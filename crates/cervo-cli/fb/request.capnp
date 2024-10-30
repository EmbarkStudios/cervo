@0xc2577ac7dec0b7dc;

struct NamedFloatList {
  name   @0 :Text;
  shape  @1 :List(UInt16);
  values @2 :List(Float32);
}

struct DataInstance {
   identity    @0 :UInt64;
   dataLists @1 :List(NamedFloatList);
}

struct Response {
  seq  @0 :UInt64;
  data @1 :List(DataInstance);
}

struct Request {
  seq  @0 :UInt64;
  data @1 :List(DataInstance);
}
