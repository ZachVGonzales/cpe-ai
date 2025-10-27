def add (a b : Nat) := a + b

def main : IO Unit := do
  let result := add 5 3
  IO.println s!"add 5 3 = {result}"
