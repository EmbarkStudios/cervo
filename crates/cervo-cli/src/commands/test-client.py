import contextlib
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests


# The requests are expected to be in the form of a POST request with the body containing the input data.
# The encoding of the input data is expected to be in the following format:
# 1. The first byte is the batch size.
# For each batch item:
#   2. The first byte is the number of inputs
#   For each input:
#     3. The first byte is the length of the input key
#     4. The next `length` bytes are the input key as utf-8
#     5. The next 4 bytes is the byte-length of the input value as LE unsigned
#     6. The next `length` bytes are the input value as float in LE)
def build_request(batch_size, input_key_to_shape, offset):
    buffer = struct.pack("B", batch_size)

    for i in range(batch_size):
        buffer += struct.pack("B", len(input_key_to_shape))

        for key, shape in input_key_to_shape.items():
            buffer += struct.pack("B", len(key))
            buffer += key.encode("utf-8")
            shape_size = 1
            for dim in shape:
                shape_size *= dim

            buffer += struct.pack("<I", shape_size * 4)
            for i in range(shape_size):
                buffer += struct.pack("<f", random.random())

    return buffer


def parse_response(buffer) -> list[dict[str, list[float]]]:
    offset = 0
    batch_size = buffer[offset]
    offset += 1

    outputs = []

    for i in range(batch_size):
        num_outputs = buffer[offset]

        offset += 1

        output = {}

        for j in range(num_outputs):
            key_length = buffer[offset]

            offset += 1
            key = buffer[offset : offset + key_length].decode("utf-8")

            offset += key_length

            shape_size = int(struct.unpack("<I", buffer[offset : offset + 4])[0] / 4)

            offset += 4

            output[key] = list(
                struct.unpack(
                    f"<{shape_size }f", buffer[offset : offset + shape_size * 4]
                )
            )
            offset += shape_size * 4

        outputs.append(output)

    return outputs


@contextlib.contextmanager
def timed_scope(name):
    start = time.perf_counter()
    yield
    print(f"{name} took {time.perf_counter() - start} seconds")


@dataclass
class Args:
    host: str
    port: int

    batch_size: int
    input_key_to_shape: dict[str, tuple[int, ...]]

    count: int

    def argparse():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, required=True)
        parser.add_argument("--port", type=int, required=True)

        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--input-key-to-shape", type=str, nargs="+", required=True)
        parser.add_argument("--count", type=int, default=1)

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
        )


BURST = 1000


def main(args: Args):
    import queue
    import threading

    count = 0
    start = time.perf_counter()

    queue = queue.Queue()

    def consume_responses(queue):
        while True:
            response = queue.get().result()

            parse_response(response.content)
            queue.task_done()

    t = threading.Thread(target=consume_responses, args=(queue,))
    t.daemon = True
    t.start()

    with ThreadPoolExecutor() as executor:
        request = build_request(args.batch_size, args.input_key_to_shape, offset=count)
        for iters in range(args.count // (args.batch_size * BURST)):
            for _ in range(BURST):
                queue.put(
                    executor.submit(
                        requests.post, f"http://{args.host}:{args.port}/", data=request
                    )
                )

            time.sleep(0.01)
            count += args.batch_size * BURST

    queue.join()

    print(f"Total time: {time.perf_counter() - start}")
    print(f"Total count: {count}")
    print(f"Average time: {(time.perf_counter() - start) / count}")
    print(f"Average rate: {count / (time.perf_counter() - start)}")


if __name__ == "__main__":
    main(Args.argparse())
