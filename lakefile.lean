import Lake
open Lake DSL

package cpe_ai

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

@[default_target]
lean_lib CpeAi
