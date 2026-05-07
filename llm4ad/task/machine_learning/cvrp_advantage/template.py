template_program = '''
import torch

def compute_advantage(reward, load, at_the_depot, finished, loss_ema, reward_ema, epoch):
    """
    Compute advantage for REINFORCE training of CVRP (Capacitated Vehicle Routing Problem).

    In POMO training, each CVRP instance is rolled out with N different starting
    customers, producing N trajectories.  The advantage determines how strongly
    each trajectory is reinforced or suppressed:  positive advantage → increase
    trajectory probability; negative advantage → decrease it.

    You may freely use torch operations.  The returned tensor must have the same
    shape as reward.

    Args:
        reward:       (batch, pomo)  float tensor — negative of total route distance.
                       Larger (closer to 0) = better route.
        load:         (batch, pomo)  float tensor — remaining vehicle capacity at end
                       of episode, range [0, 1].  Low load means high utilization.
        at_the_depot: (batch, pomo)  bool tensor  — whether the last step was at depot.
                       Frequent depot returns imply more vehicles used.
        finished:     (batch, pomo)  bool tensor  — True if all customers were visited.
                       Unfinished trajectories are invalid (capacity constraint violated).
        loss_ema:     float — exponential moving average of recent training loss.
                       Tracks whether training is converging or stuck.
        reward_ema:   float — exponential moving average of recent mean reward.
                       Indicates current policy quality.
        epoch:        int   — current training epoch (0-based).  Small = early training,
                       large = late training.

    Returns:
        advantage:    (batch, pomo)  float tensor with same shape as reward.
    """
    advantage = reward - reward.float().mean(dim=1, keepdims=True)
    return advantage
'''

task_description = """Design an advantage function for REINFORCE training of the Capacitated Vehicle Routing Problem (CVRP).

## Background
We train a neural network policy for CVRP using POMO (Policy Optimization with Multiple Optima).  In each training batch, a batch of CVRP instances is generated.  For each instance, N different trajectories (POMO rollouts) are sampled by starting from different first customers.  The REINFORCE loss is:

    loss = -(advantage * log_probability).mean()

where advantage determines the update direction and magnitude for each trajectory.  The standard advantage is reward - mean(reward) within the same instance (batch-internal baseline).  Your job is to design a better advantage function.

## CVRP Domain Knowledge
- Each customer has a demand; the vehicle has limited capacity (normalized to 1.0).
- When the vehicle returns to the depot, its capacity is refilled to 1.0.
- A trajectory is "finished" only if ALL customers have been visited.  Unfinished trajectories mean the policy violated capacity constraints or got stuck.
- Key trade-offs: shorter total distance vs. fewer vehicles vs. higher capacity utilization.

## Input Tensor Shapes
| Name          | Shape          | Type  | Meaning                                        |
|---------------|----------------|-------|------------------------------------------------|
| reward        | (batch, pomo)  | float | Negative route distance (higher = better)      |
| load          | (batch, pomo)  | float | Final remaining capacity [0, 1]                |
| at_the_depot  | (batch, pomo)  | bool  | True if last step was at depot                 |
| finished      | (batch, pomo)  | bool  | True if all customers visited                  |
| loss_ema      | scalar         | float | Exponential moving average of recent loss      |
| reward_ema    | scalar         | float | Exponential moving average of recent reward    |
| epoch         | scalar         | int   | Current training epoch number                  |

## Real-Time Training Features (loss_ema, reward_ema, epoch)
These features let your advantage function adapt over the course of training:
- Early training (small epoch): the policy is still exploring.  Advantage should encourage diversity and be tolerant of capacity violations (unfinished trajectories).
- Mid training: gradually tighten capacity penalties and start optimizing vehicle count.
- Late training (large epoch): focus on fine-grained optimization of distance and capacity utilization.
- Use loss_ema and reward_ema to detect plateaus or instability, and adjust advantage scaling accordingly.

## Design Goals
1. Accelerate convergence: the policy should reach good solutions faster than the standard advantage.
2. Penalize infeasibility: unfinished trajectories should receive strongly negative advantage.
3. Reward capacity utilization: trajectories that use vehicle capacity efficiently should be favored.
4. Adapt to training stage: use epoch, loss_ema, and reward_ema to dynamically adjust the advantage computation.
5. Maintain numerical stability: avoid extreme advantage values that cause gradient explosion.

## Tips
- You can use torch operations: mean, std, max, min, clamp, exp, log, sigmoid, tanh, etc.
- The output shape MUST match reward: (batch, pomo).
- Consider normalizing advantage within each batch to maintain stable gradient scales.
- Penalizing unfinished trajectories heavily (e.g., advantage = -1e6) is a simple but effective baseline.
- The "number of vehicles" can be inferred from how many times at_the_depot flips from False to True."""
