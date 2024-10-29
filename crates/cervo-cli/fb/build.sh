set -euxo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

flatc -r -o "../src/commands/serve/" request.fbs response.fbs types.fbs
flatc -p -o "../python/messages" request.fbs response.fbs types.fbs



find "../src/commands/serve/" -type f -exec sed -i 's/crate::types_generated::\*/super::types_generated::{self, *}/g' {} \;
find "../src/commands/serve/" -type f -exec sed -i 's/extern crate flatbuffers;//g' {} \;
find "../src/commands/serve/" -type f -exec sed -i 's/self::flatbuffers/flatbuffers/g' {} \;
find "../src/commands/serve/" -type f -exec sed -i 's/\/\/ @generated/\/\/ @generated\n#![allow(unused)]/g' {} \;
echo "pub use cervo::*;" >> "../src/commands/serve/types_generated.rs"
