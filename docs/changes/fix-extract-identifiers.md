- **`extract` keeps property names and enum values apart from its own identifiers** (task type `extract`): a property
  named `customer.name` came back nested as `customer` → `name` (and `a` next to `a.b` raised an error, HTTP 500);
  enum values `1` and `"1"` were one option, and a value `"__missing__"` read as "not stated". `data` now follows the
  property names as given, a schema where a dotted name and a nested object give the same path is refused (422), and
  enum values that would read the same are keyed by their JSON form and come back with their type. Schemas without
  such names or values keep their option keys and answers. `basal-calibrate` reads the gold of such fields the same way.
