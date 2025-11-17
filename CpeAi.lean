import Mathlib

open scoped BigOperators

-- A minimal workspace with the constructions used in the problem.
namespace MonotoneSum

variable (f : ℝ → ℝ)

/-- Define A(t) = f(t+1) - f(f(t)). -/
def A (t : ℝ) : ℝ := f (t + 1) - f (f t)

/-- Define Sₙ(x) = ∑_{i=1}^n i · A(x+i). We index over `Finset.Icc 1 n` to match 1..n. -/
def S (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.Icc 1 n).sum (fun i => (i : ℝ) * A f (x + (i : ℝ)))

/-- The original sum `∑ i=1..n i (f(x+i+1) - f(f(x+i)))` rewrites as `S n x` via the
    definition of `A`. -/
lemma rewrite_original_sum (x : ℝ) (n : ℕ) :
    (Finset.Icc 1 n).sum (fun i => (i : ℝ) * (f (x + (i : ℝ) + 1) - f (f (x + (i : ℝ))))) =
    S f n x := by
  unfold S
  refine Finset.sum_congr rfl ?_
  intro i hi
  simp [A]

/-- From the hypothesis that the original sum is bounded independently of `x` and `n`,
    we obtain the same bound for `S n x`. -/
lemma bound_S_of_bound_original {C : ℝ}
    (h : ∀ (x : ℝ) (n : ℕ),
      |(Finset.Icc 1 n).sum (fun i => (i : ℝ) * (f (x + (i : ℝ) + 1) - f (f (x + (i : ℝ)))))| < C) :
    ∀ (x : ℝ) (n : ℕ), |S f n x| < C := by
  intro x n
  simpa [rewrite_original_sum (f := f) x n]
    using h x n

/-- A one–step decomposition of `S`:
For `n ≥ 1`, the last term can be split off, giving
`S n x = S (n-1) x + n · A (x+n)`.
This is the formal version of `Sₙ(x) - Sₙ₋₁(x) = n A(x+n)`. -/
lemma S_step (x : ℝ) {n : ℕ} (hn : 1 ≤ n) :
    S f n x = S f (n - 1) x + (n : ℝ) * A f (x + (n : ℝ)) := by
  classical
  -- Work with a standalone summand `g`.
  set g : ℕ → ℝ := fun i => (i : ℝ) * A f (x + (i : ℝ)) with hg
  -- First, identify `∑_{i∈Icc 1 n} g i` with `(∑_{i∈Ico 1 n} g i) + g n`.
  have h_decomp : (Finset.Icc 1 n).sum g = (Finset.Ico 1 n).sum g + g n := by
    have := Finset.sum_Ico_add_eq_sum_Icc (a := 1) (b := n) (f := g) hn
    simpa using this.symm
  -- Next, identify `∑_{i∈Ico 1 n} g i` with `∑_{i∈Icc 1 (n-1)} g i`.
  have h_Ico_eq_Icc : (Finset.Ico 1 n).sum g = (Finset.Icc 1 (n - 1)).sum g := by
    by_cases hne : n = 1
    · subst hne
      simp [g]
    · have hlt : 1 < n := lt_of_le_of_ne hn (Ne.symm hne)
      have h1le : 1 ≤ n - 1 := Nat.le_pred_of_lt hlt
      have hA := Finset.sum_Ico_add_eq_sum_Ico_add_one (a := 1) (b := n - 1) (f := g) h1le
      have hB := Finset.sum_Ico_add_eq_sum_Icc (a := 1) (b := n - 1) (f := g) h1le
      -- `hA : (∑ Ico 1 (n-1) g) + g (n-1) = ∑ Ico 1 n g`
      -- `hB : (∑ Ico 1 (n-1) g) + g (n-1) = ∑ Icc 1 (n-1) g`
      exact (Eq.trans hA.symm hB)
  -- Convert back to the definition of `S`.
  unfold S
  -- Replace the left-hand side and use the two identities above.
  calc
    (Finset.Icc 1 n).sum (fun i => (i : ℝ) * A f (x + (i : ℝ)))
        = (Finset.Ico 1 n).sum (fun i => (i : ℝ) * A f (x + (i : ℝ)))
          + (n : ℝ) * A f (x + (n : ℝ)) := by
          simpa [hg]
            using h_decomp
    _ = (Finset.Icc 1 (n - 1)).sum (fun i => (i : ℝ) * A f (x + (i : ℝ)))
          + (n : ℝ) * A f (x + (n : ℝ)) := by
          simpa [hg] using congrArg (fun t => t + (n : ℝ) * A f (x + (n : ℝ))) h_Ico_eq_Icc

/-- The elementary inequality `|n · A(x+n)| ≤ |S n x| + |S (n-1) x|`,
obtained from `S_step` and the triangle inequality. -/
lemma abs_nA_le_absS (x : ℝ) {n : ℕ} (hn : 1 ≤ n) :
    |(n : ℝ) * A f (x + (n : ℝ))| ≤ |S f n x| + |S f (n - 1) x| := by
  have hstep := S_step (f := f) x hn
  -- Re-express `S n - S (n-1)`
  have hdiff : S f n x - S f (n - 1) x = (n : ℝ) * A f (x + (n : ℝ)) := by
    -- (a + b) - a = b
    simpa [hstep] using
      add_sub_cancel_right (S f (n - 1) x) ((n : ℝ) * A f (x + (n : ℝ)))
  -- Triangle inequality in the form `|a - b| ≤ |a - 0| + |0 - b|` via `abs_sub_le`.
  have htri' : |S f n x - S f (n - 1) x| ≤
      |S f n x - 0| + |0 - S f (n - 1) x| := by
    simpa using abs_sub_le (S f n x) 0 (S f (n - 1) x)
  have htri : |S f n x - S f (n - 1) x| ≤ |S f n x| + |S f (n - 1) x| := by
    simpa [sub_zero, sub_eq_add_neg, abs_neg] using htri'
  have h2 := htri
  simp only [hdiff] at h2
  exact h2

/-- Combining the bound on `S` with `abs_nA_le_absS`, we get the desired
`|n · A(x+n)| < 2C`. -/
lemma abs_nA_lt_twoC {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (x : ℝ) {n : ℕ} (hn : 1 ≤ n) :
    |(n : ℝ) * A f (x + (n : ℝ))| < 2 * C := by
  have h1 : |S f n x| < C := hS x n
  have h2 : |S f (n - 1) x| < C := hS x (n - 1)
  have hle : |(n : ℝ) * A f (x + (n : ℝ))| ≤ |S f n x| + |S f (n - 1) x| :=
    abs_nA_le_absS (f := f) x hn
  have hsum_lt : |S f n x| + |S f (n - 1) x| < C + C := add_lt_add h1 h2
  exact lt_of_le_of_lt hle (by simpa [two_mul] using hsum_lt)

/-- Substituting `x := t - n` in `abs_nA_lt_twoC` yields `|n · A(t)| < 2C` for all
`t ∈ ℝ` and `n ≥ 1`. -/
lemma abs_nA_at_t_lt_twoC {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (t : ℝ) {n : ℕ} (hn : 1 ≤ n) :
    |(n : ℝ) * A f t| < 2 * C := by
  -- Take `x = t - n` so that `x + n = t`.
  simpa [sub_eq_add_neg, add_assoc] using
    (abs_nA_lt_twoC (f := f) (C := C) hS (x := t - (n : ℝ)) (n := n) hn)

/-- Divide by `n` and let `n → ∞` to conclude `A(t) = 0` for all `t`.
Formally: from `|n · A(t)| < 2C` for all `n ≥ 1`, deduce `A(t) = 0`. -/
lemma A_eq_zero_of_boundS {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (t : ℝ) :
    A f t = 0 := by
  -- Suppose for contradiction that `A f t ≠ 0`.
  by_contra hA
  have hapos : 0 < |A f t| := abs_pos.mpr hA
  -- Choose an `n` so large that `n > max ((2*C)/|A|) 1`.
  obtain ⟨n0, hn0⟩ := exists_nat_gt (max ((2 * C) / |A f t|) (1 : ℝ))
  -- Strengthen to a natural `n ≥ 1`.
  let n := Nat.succ n0
  have hn1 : 1 ≤ n := by
    have : 1 ≤ Nat.succ n0 := Nat.succ_le_succ (Nat.zero_le n0)
    simpa [n] using this
  -- Use the bound `|n · A(t)| < 2C`.
  have hbound := abs_nA_at_t_lt_twoC (f := f) (C := C) hS t (n := n) hn1
  -- Rewrite the left-hand side as `(n : ℝ) * |A|`.
  have hn_nonneg : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
  have habs_rewrite : |(n : ℝ) * A f t| = (n : ℝ) * |A f t| := by
    have := abs_mul (n : ℝ) (A f t)
    -- `|(n:ℝ)| = n` since `n ≥ 0`.
    have hnn : |(n : ℝ)| = (n : ℝ) := abs_of_nonneg hn_nonneg
    simpa [hnn, mul_comm] using this
  have hineq : (n : ℝ) * |A f t| < 2 * C := by simpa [habs_rewrite] using hbound
  -- From `n > (2*C)/|A|`, deduce `2*C < n * |A|`.
  have hmax_lt_n0 : (max ((2 * C) / |A f t|) (1 : ℝ)) < (n0 : ℝ) := hn0
  have hn0_lt_n : (n0 : ℝ) < (n : ℝ) := by
    -- `n = n0 + 1`, so `(n0:ℝ) < (n:ℝ)`.
    exact_mod_cast (Nat.lt_succ_self n0)
  have hdiv_lt_n : (2 * C) / |A f t| < (n : ℝ) :=
    lt_trans (lt_of_le_of_lt (le_max_left _ _) hmax_lt_n0) hn0_lt_n
  have hbig' : (2 * C) / |A f t| * |A f t| < (n : ℝ) * |A f t| :=
    (mul_lt_mul_of_pos_right hdiv_lt_n hapos)
  have hnnz : |A f t| ≠ 0 := ne_of_gt hapos
  have hleft_eq : (2 * C) / |A f t| * |A f t| = 2 * C := by
    -- rewrite as `(2*C) * |A|⁻¹ * |A|` and cancel
    simpa [div_eq_mul_inv, hnnz, mul_comm, mul_left_comm, mul_assoc]
      using (show (2 * C) / |A f t| * |A f t| = 2 * C from by
        -- use `simp` directly
        simpa [div_eq_mul_inv, hnnz, mul_comm, mul_left_comm, mul_assoc])
  have hbig : 2 * C < (n : ℝ) * |A f t| := by
    simpa [hleft_eq] using hbig'
  -- Contradiction with `hineq`.
  have hcontr : ¬ (2 * C) < (2 * C) := lt_irrefl _
  exact hcontr (lt_trans hbig hineq)

/-- As a consequence, we have the functional equation `f (f t) = f (t + 1)` for all `t`. -/
lemma comp_iter_eq_shift {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (t : ℝ) :
    f (f t) = f (t + 1) := by
  have hA0 : A f t = 0 := A_eq_zero_of_boundS (f := f) (C := C) hS t
  have : f (t + 1) = f (f t) := (sub_eq_zero.mp hA0)
  simpa using this.symm

/-- If `f` is constant, say `f x = b` for all `x`, then the equation
`f (f x) = f (x + 1)` holds trivially. -/
lemma comp_iter_eq_shift_of_const {f : ℝ → ℝ} (b : ℝ)
    (hconst : ∀ x, f x = b) (x : ℝ) :
    f (f x) = f (x + 1) := by
  have hf : f = fun _ : ℝ => b := funext hconst
  subst hf
  simp

/-- Assume `f` is non-constant and monotone (in the usual problem sense: either
strictly increasing or strictly decreasing). Then indeed `f` is either strictly
increasing or strictly decreasing. This lemma simply records that assumption. -/
lemma nonconstant_monotone_then_strict {f : ℝ → ℝ}
    (hmono : StrictMono f ∨ StrictAnti f) (hnonconst : ∃ x y, f x ≠ f y) :
    StrictMono f ∨ StrictAnti f :=
  hmono

/-- If `f` is strictly decreasing, then `x ↦ f (f x)` is strictly increasing while
`x ↦ f (x+1)` is strictly decreasing. Hence they cannot be equal. This rules out the
case `StrictAnti f` once we know `f (f x) = f (x+1)` for all `x`. -/
lemma no_strictAnti_of_comp_iter_eq_shift {f : ℝ → ℝ}
    (hanti : StrictAnti f) (h : ∀ t, f (f t) = f (t + 1)) : False := by
  -- `f ∘ f` is strictly increasing.
  have h_mono : StrictMono (fun t : ℝ => f (f t)) := by
    intro x y hxy
    -- from `x < y` we get `f y < f x`, and applying `f` again reverses the inequality once more
    -- so `f (f x) < f (f y)`.
    exact hanti (hanti hxy)
  -- `x ↦ f (x+1)` is strictly decreasing.
  have h_anti : StrictAnti (fun t : ℝ => f (t + 1)) := by
    intro x y hxy
    have hx1 : x + 1 < y + 1 := by
      simpa using add_lt_add_right hxy (1 : ℝ)
    exact hanti hx1
  -- Using the equality of the two functions, we get a contradiction from asymmetry of `<`.
  let g := fun t : ℝ => f (f t)
  let hsh := fun t : ℝ => f (t + 1)
  have heq : g = hsh := funext h
  have hlt1 : g 0 < g 1 := h_mono (show (0 : ℝ) < 1 from zero_lt_one)
  -- Transport this inequality along pointwise equalities at `0` and `1`.
  have heq0 : g 0 = hsh 0 := congrArg (fun φ : ℝ → ℝ => φ 0) heq
  have heq1 : g 1 = hsh 1 := congrArg (fun φ : ℝ → ℝ => φ 1) heq
  have hlt1' : hsh 0 < hsh 1 := by simpa [heq0, heq1] using hlt1
  have hlt2 : hsh 1 < hsh 0 := by
    -- `h_anti` applied to `x=0`, `y=1`.
    have : (0 : ℝ) < 1 := zero_lt_one
    simpa [hsh] using (h_anti this)
  exact (lt_asymm hlt1' hlt2).elim

/-- Combining the previous contradiction with `comp_iter_eq_shift`, we can rule out
that a strictly decreasing `f` satisfies the bounded-sum hypothesis. -/
lemma no_strictAnti_of_boundS {f : ℝ → ℝ} {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (hanti : StrictAnti f) : False := by
  exact no_strictAnti_of_comp_iter_eq_shift (f := f) hanti (fun t => comp_iter_eq_shift (f := f) (C := C) hS t)

/-- Therefore `f` is strictly increasing and hence injective, provided we know that
`f` is monotone in the sense `StrictMono f ∨ StrictAnti f` and the bounded-sum
hypothesis holds. -/
lemma strictMono_and_injective_of_boundS {f : ℝ → ℝ} {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C)
    (hmono : StrictMono f ∨ StrictAnti f) :
    StrictMono f ∧ Function.Injective f := by
  -- Exclude the `StrictAnti` case and keep the `StrictMono` one.
  have hstrict : StrictMono f := by
    cases hmono with
    | inl h => exact h
    | inr h => exact (no_strictAnti_of_boundS (f := f) (C := C) hS h).elim
  exact And.intro hstrict hstrict.injective

/-- If `f` is injective and satisfies `f (f x) = f (x + 1)` for all `x`, then
`f x = x + 1` for all `x`. This is the injectivity step mentioned in the proof. -/
lemma eq_shift_of_injective_of_comp_iter_eq_shift {f : ℝ → ℝ}
    (hinj : Function.Injective f) (h : ∀ x, f (f x) = f (x + 1)) :
    ∀ x, f x = x + 1 := by
  intro x
  exact hinj (by simpa using h x)

/-- As a convenient corollary: from the bounded-sum hypothesis and injectivity of `f`,
we obtain `f x = x + 1` for all `x`. -/
lemma eq_shift_of_boundS_and_injective {f : ℝ → ℝ} {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C) (hinj : Function.Injective f) :
    ∀ x, f x = x + 1 := by
  exact eq_shift_of_injective_of_comp_iter_eq_shift (f := f) hinj
    (fun x => comp_iter_eq_shift (f := f) (C := C) hS x)

/-- Finally, under the monotonicity assumption (strictly monotone or strictly anti-monotone),
`f` must be strictly increasing and hence injective; together with `f (f x) = f (x+1)` this
forces `f x = x + 1` for all `x`. -/
lemma eq_shift_of_boundS_and_monotone {f : ℝ → ℝ} {C : ℝ}
    (hS : ∀ (x : ℝ) (n : ℕ), |S f n x| < C)
    (hmono : StrictMono f ∨ StrictAnti f) :
    ∀ x, f x = x + 1 := by
  -- Get injectivity from strict monotonicity (ruling out the anti-monotone case).
  have hstrict_inj := strictMono_and_injective_of_boundS (f := f) (C := C) hS hmono
  have hinj : Function.Injective f := hstrict_inj.2
  -- Apply injectivity to the pointwise equality `f (f x) = f (x+1)`.
  exact eq_shift_of_boundS_and_injective (f := f) (C := C) hS hinj

/-!  Auxiliary lemmas verifying that for the two explicit solutions
`f(x) = b` and `f(x) = x+1`, each term `f(x+i+1) - f(f(x+i))` vanishes,
so the whole sum is `0` and the bound holds trivially. -/

/-- For a constant function `f x = b`, each step–difference `A(t)` is `0`. -/
lemma A_const_zero (b t : ℝ) : A (fun _ : ℝ => b) t = 0 := by
  simp [A]

/-- For `f x = x+1`, each step–difference `A(t)` is `0`. -/
lemma A_shift_zero (t : ℝ) : A (fun x : ℝ => x + 1) t = 0 := by
  simp [A, add_comm, add_left_comm, add_assoc]

/-- If `A` vanishes pointwise, then every summand is zero and `S n x = 0`. -/
lemma S_eq_zero_of_A_zero {f : ℝ → ℝ} (hA : ∀ t, A f t = 0) (n : ℕ) (x : ℝ) :
    S f n x = 0 := by
  unfold S
  refine Finset.sum_eq_zero ?h
  intro i hi
  have : A f (x + (i : ℝ)) = 0 := hA _
  simp [this]

/-- If `A` vanishes pointwise, then the original sum is `0`. -/
lemma original_sum_eq_zero_of_A_zero {f : ℝ → ℝ}
    (hA : ∀ t, A f t = 0) (x : ℝ) (n : ℕ) :
    (Finset.Icc 1 n).sum (fun i => (i : ℝ) * (f (x + (i : ℝ) + 1) - f (f (x + (i : ℝ))))) = 0 := by
  have hS0 : S f n x = 0 := S_eq_zero_of_A_zero (f := f) hA n x
  simpa [hS0] using rewrite_original_sum (f := f) x n

/-- In the constant case `f x = b`, each term is `0`. -/
lemma term_zero_const (b x : ℝ) (i : ℕ) :
    ((fun _ : ℝ => b) (x + (i : ℝ) + 1) - (fun _ : ℝ => b) ((fun _ : ℝ => b) (x + (i : ℝ)))) = 0 := by
  simp

/-- In the shift case `f x = x+1`, each term is `0`. -/
lemma term_zero_shift (x : ℝ) (i : ℕ) :
    ((fun t : ℝ => t + 1) (x + (i : ℝ) + 1) - (fun t : ℝ => t + 1) ((fun t : ℝ => t + 1) (x + (i : ℝ)))) = 0 := by
  simp [add_comm, add_left_comm, add_assoc]

/-- Hence, for a constant function `f x = b`, the whole sum is `0`. -/
lemma original_sum_zero_const (b x : ℝ) (n : ℕ) :
    (Finset.Icc 1 n).sum (fun i => (i : ℝ) * ((fun _ : ℝ => b) (x + (i : ℝ) + 1) - (fun _ : ℝ => b) ((fun _ : ℝ => b) (x + (i : ℝ))))) = 0 := by
  exact original_sum_eq_zero_of_A_zero (f := fun _ : ℝ => b) (hA := A_const_zero b) x n

/-- And for `f x = x+1`, the whole sum is `0`. -/
lemma original_sum_zero_shift (x : ℝ) (n : ℕ) :
    (Finset.Icc 1 n).sum (fun i => (i : ℝ) * ((fun t : ℝ => t + 1) (x + (i : ℝ) + 1) - (fun t : ℝ => t + 1) ((fun t : ℝ => t + 1) (x + (i : ℝ))))) = 0 := by
  exact original_sum_eq_zero_of_A_zero (f := fun t : ℝ => t + 1) (hA := A_shift_zero) x n

/-- Consequently, for these two functions the strict bound `|sum| < C` holds for any `C > 0`. -/
lemma bound_original_const (b C : ℝ) (hC : 0 < C) (x : ℝ) (n : ℕ) :
    |(Finset.Icc 1 n).sum (fun i => (i : ℝ) * ((fun _ : ℝ => b) (x + (i : ℝ) + 1) - (fun _ : ℝ => b) ((fun _ : ℝ => b) (x + (i : ℝ)))))| < C := by
  simpa [original_sum_zero_const b x n, abs_zero] using hC

lemma bound_original_shift (C : ℝ) (hC : 0 < C) (x : ℝ) (n : ℕ) :
    |(Finset.Icc 1 n).sum (fun i => (i : ℝ) * ((fun t : ℝ => t + 1) (x + (i : ℝ) + 1) - (fun t : ℝ => t + 1) ((fun t : ℝ => t + 1) (x + (i : ℝ)))))| < C := by
  simpa [original_sum_zero_shift x n, abs_zero] using hC

/-- Main classification: the monotone solutions to the bounded–sum condition are exactly
all constant functions and the single nonconstant solution `f x = x + 1`.
We formalize the "monotone" hypothesis as: either `f` is constant or it is strictly
monotone (increasing) or strictly antimonotone (decreasing). -/
lemma classify_monotone_solutions {f : ℝ → ℝ}
    (hmono : (∃ b, f = fun _ : ℝ => b) ∨ StrictMono f ∨ StrictAnti f) :
    (∃ C > 0,
      ∀ (x : ℝ) (n : ℕ),
        |(Finset.Icc 1 n).sum (fun i =>
            (i : ℝ) * (f (x + (i : ℝ) + 1) - f (f (x + (i : ℝ)))))| < C) ↔
    ((∃ b, f = fun _ : ℝ => b) ∨ (∀ x, f x = x + 1)) := by
  constructor
  · intro hbound
    rcases hbound with ⟨C, hCpos, hBound⟩
    -- If `f` is constant, we are done; otherwise use strict monotonicity/anti-monotonicity.
    cases hmono with
    | inl hconst => exact Or.inl hconst
    | inr hrest =>
      cases hrest with
      | inl hsm =>
        -- convert bound on the original sum to a bound on `S` and conclude
        have hS : ∀ x n, |S f n x| < C :=
          bound_S_of_bound_original (f := f) (C := C) hBound
        exact Or.inr (eq_shift_of_boundS_and_monotone (f := f) (C := C) hS (Or.inl hsm))
      | inr hasi =>
        have hS : ∀ x n, |S f n x| < C :=
          bound_S_of_bound_original (f := f) (C := C) hBound
        exact Or.inr (eq_shift_of_boundS_and_monotone (f := f) (C := C) hS (Or.inr hasi))
  · intro hsol
    -- In either case we can take `C = 1` and the sum is identically `0`.
    cases hsol with
    | inl hconst =>
      rcases hconst with ⟨b, hf⟩
      refine ⟨1, zero_lt_one, ?_⟩
      intro x n
      subst hf
      simpa using (bound_original_const (b := b) (C := 1) zero_lt_one x n)
    | inr hshift =>
      -- `f = (· + 1)`
      have hf : f = fun t : ℝ => t + 1 := funext hshift
      refine ⟨1, zero_lt_one, ?_⟩
      intro x n
      subst hf
      simpa using (bound_original_shift (C := 1) zero_lt_one x n)

end MonotoneSum

/-- Example content kept from the initial workspace. -/
def hello : String :=
  "Hello from Lean workspace!"

#eval IO.println s!"{hello}"
