import Mathlib.LinearAlgebra.FiniteDimensional.Defs

set_option maxRecDepth 1000
set_option maxHeartbeats 3000000

universe u v
variable {R} { Vminus V W V0 V1 V2 V3 : Type u} {M' : Type v} [DivisionRing R]
variable  [AddCommGroup V] [AddCommGroup W]
variable [Module R V] [Module R W]
variable [FiniteDimensional R V] [FiniteDimensional R W]
