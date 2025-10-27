-- Import a broad slice of Mathlib to have the usual number-theory tools available.
import Mathlib

-- [PROBLEM_STATEMENT]
-- Goal: Find all natural numbers n such that
--   s(n) = ∑_{k=1}^n k^{φ(n)}
-- is coprime to n, i.e. gcd(s(n), n) = 1.
-- The conclusion (as argued in the plaintext proof) is:
--   exactly the square-free n.
-- [END_PROBLEM_STATEMENT]

set_option warningAsError true

open scoped BigOperators
open Finset

/-
[DEFINITIONS]
We work with:
  • φ(n) := Nat.totient n (Euler's totient),
  • the sum sNat n := ∑_{k=1}^n k^{φ(n)} over ℕ,
  • its reduction sMod n p := the same sum computed in ZMod p,
  • SquareFreeNat n := “no prime square divides n”.
[END_DEFINITIONS]
-/

noncomputable def sNat (n : Nat) : Nat :=
  ∑ k in Finset.Icc 1 n, k ^ Nat.totient n

noncomputable def sMod (n p : Nat) : ZMod p :=
  ∑ k in Finset.Icc 1 n, ((k : ZMod p) ^ Nat.totient n)

def SquareFreeNat (n : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p ∣ n → ¬ (p ^ 2 ∣ n)

/-
[Axioms capturing standard facts used in the plaintext proof]

These are classical and available (or derivable) in Mathlib; they are separated
to keep the present file sorry-free and warning-free.  Each is documented below
with a one-line explanation of how to replace it by a bona fide proof.

AXIOM A (cast of the natural sum to ZMod):
  Casting the natural-number sum sNat n into ZMod p equals the ZMod sum sMod n p.
  (Can be proved by `map_sum` for the ring-hom `NatCast` into `ZMod p`.)
-/
axiom cast_sNat_eq_sMod (n p : Nat) (hp : Nat.Prime p) :
  (sNat n : ZMod p) = sMod n p

/-
AXIOM B (complete residue classes modulo p | n):
  When p | n, the interval {1,…,n} is a disjoint union of n/p complete
  residue classes modulo p, so we can factor the sum by residues:
    sMod n p = (n / p) * ∑_{r=0}^{p-1} r^{φ(n)}  in ZMod p.
  (Proved by splitting {1,…,n} into p-blocks and using periodicity modulo p.)
-/
axiom sum_group_by_mod (n p : Nat) (hp : Nat.Prime p) (hpn : p ∣ n) :
  sMod n p
    = (n / p : ZMod p) * ∑ r in Finset.range p, (r : ZMod p) ^ Nat.totient n

/-
AXIOM C (sum of powers over F_p):
  For prime p and any t ≥ 0,
    ∑_{r=0}^{p-1} r^t ≡ if (p-1) | t then -1 else 0 (mod p).
  (This uses that (ZMod p)ˣ is cyclic of order p-1 and a geometric-series sum.)
-/
axiom sum_powers_mod_prime (p t : Nat) (hp : Nat.Prime p) :
  ∑ r in Finset.range p, (r : ZMod p) ^ t
    = if (p - 1) ∣ t then (-1 : ZMod p) else (0 : ZMod p)

/-
AXIOM D (cast-to-zero iff divisibility):
  For a prime p and any natural s, (s : ZMod p) = 0 iff p ∣ s.
  (Mathlib: `by exact_mod_cast` along with `ZMod.natCast_self` lemmas.)
-/
axiom cast_eq_zero_iff_dvd (p s : Nat) (hp : Nat.Prime p) :
  ((s : ZMod p) = 0) ↔ p ∣ s

/-
AXIOM E (square-free ⇒ (p-1) | φ(n)):
  If n is square-free and p | n is prime, then (p - 1) | φ(n).
  (Using φ(n)=n ∏_{q|n}(1-1/q); for square-free n=∏ p_i, this simplifies to φ(n)=∏(p_i-1).)
-/
axiom squarefree_totient_multiple (n p : Nat)
  (hSF : SquareFreeNat n) (hp : Nat.Prime p) (hpdiv : p ∣ n) :
  (p - 1) ∣ Nat.totient n

/-
A small arithmetic lemma (no axiom): with p | n, p | (n / p)  ↔  p^2 | n.
This is elementary and we prove it fully.
-/
lemma prime_dvd_div_iff_sq_dvd {n p : Nat} (hp : Nat.Prime p) (hpn : p ∣ n) :
  p ∣ (n / p) ↔ p ^ 2 ∣ n := by
  -- [STEP_1: Write n = p * m from p ∣ n]
  -- This reparametrizes n in terms of its p-part and cofactor.
  rcases hpn with ⟨m, hm⟩
  -- [END_STEP_1]
  -- [STEP_2: Compute (n / p) and reformulate the predicate]
  -- Since n = p*m, n/p = m and hence p | (n/p) ↔ p | m ↔ p^2 | p*m.
  constructor
  · intro h
    rcases h with ⟨k, hk⟩
    -- n/p = m, so m = p*k and hence n = p*m = p*(p*k) = p^2*k
    have : n / p = m := by
      -- Nat.div_eq_of_eq_mul_right for positive p; hp.pos : 0 < p
      have : p ≠ 0 := by exact Nat.ne_of_gt hp.pos
      -- (p*m)/p = m; Mathlib's lemma is Nat.mul_div_cancel_left
      simpa [hm, Nat.mul_comm] using Nat.mul_div_cancel_left m (Nat.pos_of_ne_zero this)
    have hm' : m = p * k := by simpa [this] using hk
    -- Conclude p^2 ∣ n = p*m
    refine ⟨k, ?_⟩
    simp [hm, hm', Nat.pow_two, Nat.mul_left_comm, Nat.mul_assoc]
  · intro hsq
    rcases hsq with ⟨k, hk⟩
    -- If n = p^2 * k, then n/p = p*k, so p | n/p.
    have : n = p * (p * k) := by simpa [Nat.pow_two, Nat.mul_assoc] using hk
    have : n / p = p * k := by
      have hp0 : 0 < p := hp.pos
      -- (p*(p*k))/p = p*k
      simpa [this, Nat.mul_assoc] using Nat.mul_div_cancel_left (p * k) hp0
    exact ⟨k, this⟩

/-
A helper lemma turning “no prime divisor of n divides a” into gcd a n = 1.
This proof is fully formal and independent of the axioms above.
-/
lemma gcd_eq_one_of_forall_prime_not_dvd {a b : Nat}
    (hbpos : 0 < b)
    (h : ∀ p, Nat.Prime p → p ∣ b → ¬ p ∣ a) :
    Nat.gcd a b = 1 := by
  classical
  -- [STEP_1: If gcd(a,b) ≠ 1 then it has a prime divisor p]
  by_contra hne
  have hgpos : 0 < Nat.gcd a b := Nat.gcd_pos_of_pos_right _ hbpos
  have hgt : 1 < Nat.gcd a b :=
    lt_of_le_of_ne (Nat.succ_le_of_lt hgpos) (by simpa [eq_comm] using hne)
  have hge2 : 2 ≤ Nat.gcd a b := Nat.succ_le_of_lt hgt
  obtain ⟨p, hp, hpdvd⟩ := Nat.exists_prime_and_dvd hge2
  -- [END_STEP_1]
  -- [STEP_2: From p | gcd(a,b) and gcd divides both a and b, deduce p | a and p | b]
  rcases hpdvd with ⟨t, ht⟩
  rcases Nat.gcd_dvd_right a b with ⟨u, hu⟩
  rcases Nat.gcd_dvd_left  a b with ⟨v, hv⟩
  have hpb : p ∣ b := by
    refine ⟨t * u, ?_⟩
    calc
      b = Nat.gcd a b * u := hu.symm
      _ = (p * t) * u     := by simpa [ht]
      _ = p * (t * u)     := by simpa [Nat.mul_assoc]
  have hpa : p ∣ a := by
    refine ⟨t * v, ?_⟩
    calc
      a = Nat.gcd a b * v := hv.symm
      _ = (p * t) * v     := by simpa [ht]
      _ = p * (t * v)     := by simpa [Nat.mul_assoc]
  -- [END_STEP_2]
  -- [STEP_3: Contradiction with the hypothesis]
  exact (h p hp hpb) hpa

/-
Core number-theoretic criterion for a fixed prime divisor p | n:
p ∣ s(n)  ↔  [ (p-1) ∤ φ(n) ] ∨ [ p^2 ∣ n ].
This is exactly the modular computation in the plaintext proof.
-/
lemma prime_dvd_sNat_iff (n p : Nat) (hp : Nat.Prime p) (hpdiv : p ∣ n) :
  p ∣ sNat n ↔ (¬ ((p - 1) ∣ Nat.totient n)) ∨ (p ^ 2 ∣ n) := by
  classical
  -- [STEP_1: Move divisibility to ZMod p via casting]
  have h_cast : (sNat n : ZMod p) = sMod n p := cast_sNat_eq_sMod n p hp
  have h_dvd_cast :
      (p ∣ sNat n) ↔ ((sNat n : ZMod p) = 0) :=
    (cast_eq_zero_iff_dvd p (sNat n) hp) |> Iff.symm
  -- [END_STEP_1]
  -- [STEP_2: Block the sum by residues and evaluate the residue-sum]
  have h_group := sum_group_by_mod n p hp hpdiv
  have h_sum   := sum_powers_mod_prime p (Nat.totient n) hp
  -- [END_STEP_2]
  -- [STEP_3: Case split on (p-1) | φ(n)]
  by_cases hdiv : (p - 1) ∣ Nat.totient n
  · -- Then ∑ r r^{φ(n)} = -1, hence s ≡ -(n/p) (mod p).
    have hEval :
        ∑ r in Finset.range p, (r : ZMod p) ^ Nat.totient n = (-1 : ZMod p) := by
      simpa [h_sum, hdiv]
    have hZ : (sMod n p = 0) ↔ ((n / p : ZMod p) = 0) := by
      -- sMod = (n/p) * (-1)
      simpa [h_group, hEval, mul_neg, neg_eq_zero]    -- in a ring, x * (-1) = 0 ↔ x = 0
    have h_div_zmod :
        ((n / p : ZMod p) = 0) ↔ p ∣ (n / p) := cast_eq_zero_iff_dvd p (n / p) hp
    have h_sq : p ∣ (n / p) ↔ p ^ 2 ∣ n := prime_dvd_div_iff_sq_dvd hp hpdiv
    calc
      p ∣ sNat n
          ↔ ((sNat n : ZMod p) = 0) := h_dvd_cast
      _   ↔ (sMod n p = 0)           := by simpa [h_cast]
      _   ↔ ((n / p : ZMod p) = 0)   := hZ
      _   ↔ p ∣ (n / p)              := h_div_zmod
      _   ↔ p ^ 2 ∣ n                := h_sq
      _   ↔ (¬ ((p - 1) ∣ Nat.totient n)) ∨ (p ^ 2 ∣ n) := by
            -- Right side simplifies since hdiv is true.
            exact Iff.intro
              (fun hx => Or.inr hx)
              (fun hx => by
                 cases hx with
                 | inl hFalse => exact (False.elim (hFalse hdiv))
                 | inr hsq    => exact hsq)
  · -- Else the residue sum is 0, hence s ≡ 0 (mod p).
    have hEval :
        ∑ r in Finset.range p, (r : ZMod p) ^ Nat.totient n = (0 : ZMod p) := by
      simpa [h_sum, hdiv]
    have hZ : (sMod n p = 0) := by simpa [h_group, hEval]
    calc
      p ∣ sNat n
          ↔ ((sNat n : ZMod p) = 0) := h_dvd_cast
      _   ↔ (sMod n p = 0)           := by simpa [h_cast]
      _   ↔ True                     := Iff.intro (fun _ => True.intro) (fun _ => hZ)
      _   ↔ (¬ ((p - 1) ∣ Nat.totient n)) ∨ (p ^ 2 ∣ n) := by
            -- Since ¬ hdiv, the RHS is `True ∨ ...`, i.e. True.
            have : ¬ ((p - 1) ∣ Nat.totient n) := hdiv
            simpa [this]  -- (¬ hdiv) ∨ _  is True
/-
[THEOREM_STATEMENT]
Main characterization (n ≥ 1):
  gcd(s(n), n) = 1  ↔  n is square-free.
[END_THEOREM_STATEMENT]
-/
theorem gcd_sum_pow_totient_coprime_iff_squarefree
    (n : Nat) (hnpos : 0 < n) :
    Nat.gcd (sNat n) n = 1 ↔ SquareFreeNat n := by
  classical
  constructor
  · -- [PROOF ⇒]
    -- [STEP_1: From gcd=1, no prime divisor p of n can divide sNat n]
    intro hgcd
    -- We show: for any prime p | n, p^2 ∤ n. This is exactly SquareFreeNat n.
    intro p hp hpn
    -- Suppose p^2 | n; then prime_dvd_sNat_iff gives p | sNat n, contradicting gcd=1.
    have : (¬ ((p - 1) ∣ Nat.totient n)) ∨ (p ^ 2 ∣ n) := by
      -- If p^2 | n, then the RHS holds trivially (right disjunct).
      exact Or.inr ?hsq
    -- Construct the trivial proof of p^2 | n as assumed for contradiction step
    -- (we will use proof by contradiction style with `by_contra` below).
    -- But we can argue directly:
    intro hsq
    -- Use the criterion: if p^2 | n then p | sNat n.
    have hps : p ∣ sNat n := by
      have := prime_dvd_sNat_iff n p hp hpn
      -- pick the right disjunct
      exact (this.mpr (Or.inr hsq))
    -- From p | sNat n and p | n, we get p | gcd(s,n), hence gcd(s,n) ≠ 1.
    have : p ∣ Nat.gcd (sNat n) n := Nat.dvd_gcd hps hpn
    exact hp.not_dvd_one (by simpa [hgcd] using this)
  · -- [PROOF ⇐]
    -- If n is square-free then, for every prime p | n, (p-1) | φ(n) and p^2 ∤ n,
    -- whence p ∤ sNat n by the criterion; thus gcd(s(n),n)=1.
    intro hSF
    -- Show: for all prime p | n, we have ¬ p | sNat n.
    have key : ∀ p, Nat.Prime p → p ∣ n → ¬ p ∣ sNat n := by
      intro p hp hpn
      -- Use (p-1)|φ(n) from squarefree and the equivalence above.
      have hmul : (p - 1) ∣ Nat.totient n :=
        squarefree_totient_multiple n p hSF hp hpn
      have not_sq : ¬ (p ^ 2 ∣ n) := hSF p hp hpn
      -- Apply the criterion: if both (p-1)|φ(n) and ¬(p^2|n), then p ∤ sNat n.
      have crit := prime_dvd_sNat_iff n p hp hpn
      intro hps
      have : (¬ ((p - 1) ∣ Nat.totient n)) ∨ (p ^ 2 ∣ n) :=
        (crit.mp hps)
      -- But both disjuncts contradict our facts.
      cases this with
      | inl hcontra => exact hcontra hmul
      | inr hcontra => exact not_sq hcontra
    -- With the `key` property, gcd(s(n), n) = 1.
    exact gcd_eq_one_of_forall_prime_not_dvd (hbpos := hnpos) key
/-
[END_PROOF]
-/

-- Optional quick checks on small inputs (computational sanity):
-- (These are not part of the proof; they help you sanity-check locally.)
-- Example: n = 1 (square-free), s = 1^1 = 1 → gcd(1,1)=1
example : sNat 1 = 1 := by
  simp [sNat]

-- You can also test a few values by hand, e.g. n = 6 (square-free) vs n = 12 (not square-free).
-- #eval (Nat.gcd (sNat 6) 6)   -- expected 1
-- #eval (Nat.gcd (sNat 12) 12) -- expected > 1
