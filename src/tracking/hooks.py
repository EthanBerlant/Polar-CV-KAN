from collections import defaultdict

import torch
from torch import nn


class DebugHook:
    def __init__(self, log_frequency=10):
        self.log_frequency = log_frequency
        self.step = 0
        self.stats = defaultdict(list)
        self.handlers = []

    def register(self, model: nn.Module):
        """Attaches hooks to all leaf modules."""
        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue

            # Register forward hook for activations
            self.handlers.append(module.register_forward_hook(self._make_forward_hook(name)))

            # Register backward hook for gradients
            # Note: register_full_backward_hook is preferred in modern pytorch
            # DISABLED: Causing in-place operation errors with ReLU/autograd
            # self.handlers.append(
            #    module.register_full_backward_hook(self._make_backward_hook(name))
            # )

    def _make_forward_hook(self, name):
        def hook(module, input, output):
            if self.step % self.log_frequency != 0:
                return

            # Handle tuple outputs (e.g. RNNs, Transformers)
            if isinstance(output, tuple):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                return

            # Detach to avoid interfering with autograd/inplace ops
            out_detached = output.detach()

            with torch.no_grad():
                # Magnitude statistics
                if torch.is_complex(out_detached):
                    mag = torch.abs(out_detached)
                    phase = torch.angle(out_detached)

                    self.stats[f"{name}.mag_mean"].append(mag.mean().item())
                    self.stats[f"{name}.mag_std"].append(mag.std().item())
                    self.stats[f"{name}.phase_std"].append(
                        phase.std().item()
                    )  # Circular std approx

                    # Collapsed neurons (mag near 0)
                    dead = (mag < 1e-6).float().mean().item()
                    self.stats[f"{name}.dead_pct"].append(dead)
                else:
                    self.stats[f"{name}.act_mean"].append(out_detached.mean().item())
                    self.stats[f"{name}.act_std"].append(out_detached.std().item())

        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if self.step % self.log_frequency != 0:
                return

            # grad_output is a tuple (grad_wrt_output,)
            if not grad_output or not isinstance(grad_output[0], torch.Tensor):
                return

            grad = grad_output[0]
            with torch.no_grad():
                norm = torch.norm(grad).item()
                self.stats[f"{name}.grad_norm"].append(norm)

        return hook

    def step_counters(self):
        self.step += 1

    def get_latest_stats(self):
        """Returns a dict of the latest recorded stats for printing/logging."""
        latest = {}
        for k, v in self.stats.items():
            if v:
                latest[k] = v[-1]
        return latest

    def clear(self):
        self.handlers.clear()
