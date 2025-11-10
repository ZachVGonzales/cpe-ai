import Lake
open Lake DSL

package cpe_ai {
  -- add configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

-- Define a library target
lean_lib CpeAi {
  -- add library configuration options here
}
