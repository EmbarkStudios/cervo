"""This is an automatically generated stub for `request.capnp`."""
import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "request.capnp"))
module = capnp.load(module_file)  # pylint: disable=no-member
NamedFloatList = module.NamedFloatList
NamedFloatListBuilder = NamedFloatList
NamedFloatListReader = NamedFloatList
DataInstance = module.DataInstance
DataInstanceBuilder = DataInstance
DataInstanceReader = DataInstance
Response = module.Response
ResponseBuilder = Response
ResponseReader = Response
Request = module.Request
RequestBuilder = Request
RequestReader = Request
