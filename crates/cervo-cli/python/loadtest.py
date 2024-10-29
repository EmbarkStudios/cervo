#!python3.11
import contextlib
import copyreg
import io
import multiprocessing
import queue
import random
import statistics
import struct
import threading
import time
# no judging
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor as TPE
from dataclasses import dataclass
# from multiprocessing import Pool as ThreadPoolExecutor
from multiprocessing import reduction

multiprocessing.set_start_method("fork", True)

PROCS = 240
SCALE = 1
ctx = multiprocessing.get_context()
barrier = ctx.Barrier(PROCS + 1)
import request_capnp


def build_request(batch_size, input_key_to_shape, offset):
    message = request_capnp.Request.new_message()
    message.seq = offset

    batch_members = message.init("data", batch_size)
    for i in range(batch_size):
        batch_member = batch_members[i]
        batch_member.identity = i

        lists = batch_member.init("dataLists", len(input_key_to_shape))

        for i, (key, shape) in enumerate(input_key_to_shape.items()):
            data = lists[i]
            data.name = key
            shape_data = data.init("shape", len(shape))
            shape_prod = 1

            for idx, dim in enumerate(shape):
                shape_prod *= dim
                shape_data[idx] = dim

            items = data.init("values", shape_prod)

            for i in range(shape_prod):
                items[i] = random.random()

    return message.to_bytes_packed()


@dataclass
class Args:
    host: str
    port: int

    batch_size: int
    input_key_to_shape: dict[str, tuple[int, ...]]

    count: int
    models_per_proc: int

    def argparse():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=11223)

        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--input-key-to-shape", type=str, nargs="+", required=True)
        parser.add_argument("--count", type=int, default=1)
        parser.add_argument("--models-per-proc", type=int, default=1)
        parsed = parser.parse_args()

        input_key_to_shape = {}
        for pair in parsed.input_key_to_shape:
            key, shape = pair.split(":")
            input_key_to_shape[key] = tuple(map(int, shape.split(",")))

        return Args(
            host=parsed.host,
            port=parsed.port,
            batch_size=parsed.batch_size,
            input_key_to_shape=input_key_to_shape,
            count=parsed.count,
            models_per_proc=parsed.models_per_proc,
        )


def submitter(address, batch_size, input_key_to_shape, count, model_count):
    import requests

    total_count = 0
    request = [
        build_request(n, input_key_to_shape, offset=count)
        for n in range(1, batch_size + 1)
    ]

    session = requests.Session()
    times = []
    doit = lambda r: session.post(address, data=r)
    total_requests = 0
    sleep_target = 1 / (15 * model_count) - 0.001

    barrier.wait()
    start = time.perf_counter()

    try:
        for it in range(count * model_count):
            rs = time.perf_counter()
            doit(request[it % batch_size])
            re = time.perf_counter()
            elapsed = re - rs
            times.append(elapsed * 1000)
            total_count += (it % batch_size) + 1
            total_requests += 1
            time.sleep(max(0, sleep_target - (time.perf_counter() - rs)))
    except e:
        print(e, flush=True)

    end = time.perf_counter()
    elapsed = end - start
    return (elapsed, total_count, times, total_requests / elapsed)


def main(args: Args):
    address = [f"http://{args.host}:{args.port + i}/" for i in range(4)]

    with ThreadPoolExecutor(PROCS, mp_context=ctx) as executor:
        futs = [
            executor.submit(
                submitter,
                address[i % len(address)],
                args.batch_size,
                args.input_key_to_shape,
                args.count
                // (args.batch_size * (PROCS * SCALE) * args.models_per_proc),
                args.models_per_proc,
            )
            for i in range(PROCS)
        ]

        for i in range(100):
            time.sleep(0.1)
            if barrier.n_waiting == PROCS:
                barrier.wait()
                break

        start = time.perf_counter()
        all_times = []
        es = []
        requests_per_second = []
        count = 0
        for f in futs:
            elapsed, total, times, rps = f.result()
            requests_per_second.append(rps)
            es.append(elapsed)
            count += total
            all_times.extend(times)

        end = time.perf_counter()

    rate = count / (end - start)
    print(f"Total time: {end - start}")
    print(f"Total count: {count}")
    print(f"Average time: {(end - start) / count}")
    print(f"Average rate: {rate}")

    dist = statistics.NormalDist.from_samples(all_times)
    print(
        f"Distribution: mean={dist.mean:.1f} ms, mode={dist.mode:.1f}ms, stdev={dist.stdev:.1f}ms^2"
    )
    print(
        f"              median={dist.median:.1f} ms, max={max(all_times):.1f}ms, min={min(all_times):.1f}ms"
    )
    batch_size = 1.5

    request = build_request(1, args.input_key_to_shape, 0) + build_request(
        2, args.input_key_to_shape, 0
    )

    print(f"Emulated {PROCS * SCALE} servers sending data every frame.")
    print(
        f"At {args.models_per_proc * batch_size} agents per server, 15 Hz, with {args.models_per_proc} different brains, we expected {args.models_per_proc * batch_size * PROCS * SCALE * 15} samples per second to be handled."
    )
    print(
        f"Each message was {len(request) / 2} bytes, which means we sent {rate * len(request) / 2 * 8 / 1_000_000:.1f} MBit/second"
    )
    print(
        f"Average rps: {sum(requests_per_second)/ len(requests_per_second)}, {max(requests_per_second)}, {min(requests_per_second)}"
    )
    print(f"Average es: {sum(es)/ len(es)}, {max(es)}, {min(es)}")


if __name__ == "__main__":
    main(Args.argparse())
