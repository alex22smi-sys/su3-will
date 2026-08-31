![su3-will](assets/su3-will-diagram.png)
# su3-will

**SU(3) Gauge-Covariant Neural Networks with a Mathematical Model of Will**

The first GNN architecture with *provable* SU(3) gauge covariance that models human communication as fundamental forces.

## Physics = Psychology

We decompose communication into 3 forces:

| Force | Physics | Human | Code |
| --- | --- | --- | --- |
| **Gluon** | Strong Force | Touch, Presence | `U @ Z_gluon` |
| **Real Photon** | EM Radiation | Words, Sound | `U @ Z_real * sigmoid(edge_attr)` |
| **Virtual Photon** | Thought | Interpretation | `U @ Z_photon * exp(i*phase(dst))` |

**Will** = `softplus(mass * anticipation + will_power - threshold)`  
Accumulated overcoming. The network learns when to choose truth over thought.

**Gravity** = `mass_dst / (mass_src + mass_dst)`  
Hierarchy emerges from physics, not heuristics.

## Results v3.4

Strict gauge covariance achieved:
