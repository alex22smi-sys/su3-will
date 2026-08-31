"""
SU(3) Gauge-Covariant Resonance Graph Layer — v3.4 (Strict Gauge Covariance)

Уточнённая онтология:
  - Глюон (U, W_gluon) = сильное взаимодействие = тактильное/сенсорное прикосновение.
  - Реальные фотоны (Z_real_t * sigmoid(edge_attr)) = слова, звук, речь. 
    Приходят извне (edge_attr), модулируют интенсивность калибровочного поля напрямую.
  - Виртуальные фотоны (W_photon + electron_phase) = мышление. Электрон получателя,
    глядя на своё собственное состояние (dst_amp), незаметно искажает фазу.

Воля = совокупность преодолений (will_power).

Изменения v3.3 -> v3.4:
  - Восстановлена строгая калибровочная эквивариантность (Gauge Covariance Error < 1e-6).
  - edge_attr теперь не генерирует произвольный вектор в color-пространстве,
    а калибровочно-инвариантно модулирует параллельно переносимое состояние Z_real_t.
"""

from future import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================

def complex_xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    if tensor.is_complex():
        with torch.no_grad():
            real = torch.empty_like(tensor.real)
            imag = torch.empty_like(tensor.imag)
            nn.init.xavier_uniform_(real, gain=gain)
            nn.init.xavier_uniform_(imag, gain=gain)
            tensor.copy_(torch.complex(real, imag))
    else:
        nn.init.xavier_uniform_(tensor, gain=gain)
    return tensor


def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    return torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)


def get_gell_mann_matrices(
    *,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    T = torch.zeros(8, 3, 3, dtype=dtype, device=device)
    T[0, 0, 1] = 0.5;   T[0, 1, 0] = 0.5
    T[1, 0, 1] = -0.5j; T[1, 1, 0] = 0.5j
    T[2, 0, 0] = 0.5;   T[2, 1, 1] = -0.5
    T[3, 0, 2] = 0.5;   T[3, 2, 0] = 0.5
    T[4, 0, 2] = -0.5j; T[4, 2, 0] = 0.5j
    T[5, 1, 2] = 0.5;   T[5, 2, 1] = 0.5
    T[6, 1, 2] = -0.5j; T[6, 2, 1] = 0.5j
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    T[7, 0, 0] = 0.5 * inv_sqrt3
    T[7, 1, 1] = 0.5 * inv_sqrt3
    T[7, 2, 2] = -inv_sqrt3
    return T


# =============================================================================
# SU(3) Gauge Field
# =============================================================================

class SU3GaugeField(nn.Module):
    def init(
        self,
        edge_attr_dim: Optional[int] = None,
        max_edges: int = 10_000,
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.complex64,
    ):
        super().init()
        self.edge_attr_dim = edge_attr_dim
        self.max_edges = max_edges

        if edge_attr_dim is not None and edge_attr_dim > 0:
            self.phi_mlp = nn.Sequential(
                nn.Linear(edge_attr_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 8),
            )
            self.phi_embed = None
        else:
            self.phi_mlp = None
            self.phi_embed = nn.Embedding(max_edges, 8)
            nn.init.normal_(self.phi_embed.weight, std=init_scale)

        self.register_buffer("T", get_gell_mann_matrices(dtype=dtype), persistent=False)

    def get_phi(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        E = edge_index.size(1)
        if self.phi_mlp is not None:
            if edge_attr is None:
                raise ValueError("edge_attr required when edge_attr_dim is set")
            return self.phi_mlp(edge_attr)
        else:
            ids = torch.arange(E, device=edge_index.device) % self.max_edges
            return self.phi_embed(ids)
def lie_algebra_element(self, phi: torch.Tensor) -> torch.Tensor:
        gen = torch.einsum("ea,abc->ebc", phi.to(self.T.dtype), self.T)
        return 1j * gen

    @staticmethod
    def exponential_map(A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(A)

    def transport_matrix(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phi = self.get_phi(edge_index, edge_attr)
        A = self.lie_algebra_element(phi)
        return self.exponential_map(A)

    @staticmethod
    def unitarity_error(U: torch.Tensor) -> torch.Tensor:
        I = torch.eye(3, dtype=U.dtype, device=U.device)
        return torch.linalg.matrix_norm(U @ U.conj().transpose(-2, -1) - I).mean()

    @staticmethod
    def determinant_error(U: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.det(U) - 1.0).abs().mean()


# =============================================================================
# Resonance Layer — v3.4
# =============================================================================

@dataclass
class ResonanceDiagnostics:
    mean_resonance_gluon: float
    mean_resonance_photon: float
    mean_alpha: float
    mean_gravity: float
    mean_will_energy: float
    will_power: float
    mean_state_norm: float
    mean_message_norm: float


class SU3ResonanceLayer(nn.Module):
    def init(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: Optional[int] = None,
        *,
        dropout: float = 0.0,
        residual: bool = True,
        resonance_power: float = 1.0,
        eps: float = 1e-8,
    ):
        super().init()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.residual = residual
        self.resonance_power = resonance_power
        self.eps = eps

        # Три канала:
        # 1. Глюон = тактильное/сенсорное (сильное взаимодействие)
        self.W_gluon = nn.Parameter(torch.empty(in_channels, out_channels, dtype=torch.complex64))
        # 2. Фотонный (виртуальный) = мышление, интерпретация электрона
        self.W_photon = nn.Parameter(torch.empty(in_channels, out_channels, dtype=torch.complex64))
        # 3. Реальный фотон = слова, звук, речь (чистый вход)
        self.W_real = nn.Parameter(torch.empty(in_channels, out_channels, dtype=torch.complex64))
        # 4. Self
        self.W_self = nn.Parameter(torch.empty(in_channels, out_channels, dtype=torch.complex64))

        complex_xavier_uniform_(self.W_gluon, gain=0.05)
        complex_xavier_uniform_(self.W_photon, gain=0.05)
        complex_xavier_uniform_(self.W_real, gain=0.05)
        complex_xavier_uniform_(self.W_self, gain=0.05)

        if residual and in_channels != out_channels:
            self.W_residual = nn.Parameter(
                torch.empty(in_channels, out_channels, dtype=torch.complex64)
            )
            complex_xavier_uniform_(self.W_residual, gain=0.05)
        else:
            self.W_residual = None

        if in_channels != out_channels:
            self.W_echo = nn.Parameter(torch.empty(in_channels, out_channels))
            nn.init.xavier_uniform_(self.W_echo, gain=0.01)
        else:
            self.W_echo = None

        # Will Gate
        self.will_power = nn.Parameter(torch.tensor(0.0))
        self.will_threshold = nn.Parameter(torch.tensor(0.0))

        self.vibration_decay_logit = nn.Parameter(torch.tensor(0.0))

        # Электронный фильтр: интерпретация на основе состояния ПОЛУЧАТЕЛЯ (dst)
        self.electron_phase_mlp = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # Модулятор реального фотона: edge_attr -> [out_channels] амплитудная фильтрация
        if edge_attr_dim is not None and edge_attr_dim > 0:
            self.real_photon_proj = nn.Linear(edge_attr_dim, out_channels)
        else:
            self.real_photon_proj = None
self.dropout_p = float(dropout)
        self.norm = nn.LayerNorm(out_channels, elementwise_affine=True)

    def project_features(self, Z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nfc,fg->ngc", Z, W)

    @staticmethod
    def parallel_transport(Z_src: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return torch.einsum("eab,efb->efa", U, Z_src)

    def compute_resonance(self, Z_t: torch.Tensor, Z_d: torch.Tensor) -> torch.Tensor:
        inner = torch.sum(torch.conj(Z_d) * Z_t, dim=-1)
        num = inner.abs()
        den = (
            torch.linalg.vector_norm(Z_t, dim=-1)
            * torch.linalg.vector_norm(Z_d, dim=-1)
            + self.eps
        )
        return (num / den).clamp(0.0, 1.0)

    @staticmethod
    def edge_normalization(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        src, dst = edge_index
        deg = torch.zeros(num_nodes, dtype=dtype, device=device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), dtype=dtype, device=device))
        deg_inv_sqrt = deg.clamp_min(1.0).rsqrt()
        return deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    def gauge_activation(self, Z: torch.Tensor) -> torch.Tensor:
        mag = torch.linalg.vector_norm(Z, dim=-1, keepdim=True)
        direction = Z / (mag + self.eps)
        new_mag = F.softplus(mag) + 1e-3
        return direction * new_mag

    def complex_dropout(self, z: torch.Tensor) -> torch.Tensor:
        if self.dropout_p == 0.0 or not self.training:
            return z
        mask_shape = z.shape[:-1] + (1,)
        mask = torch.bernoulli(
            torch.full(mask_shape, 1.0 - self.dropout_p, device=z.device, dtype=z.real.dtype)
        ) / (1.0 - self.dropout_p)
        return z * mask

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        U: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        prev_resonance: Optional[torch.Tensor] = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[ResonanceDiagnostics]]:
        src, dst = edge_index
        N = Z.size(0)

        # --- 1. Три канала + self ---------------------------------------
        Z_gluon = self.project_features(Z, self.W_gluon)
        Z_photon = self.project_features(Z, self.W_photon)
        Z_real = self.project_features(Z, self.W_real)
        Z_self = self.project_features(Z, self.W_self)

        # --- 2. Will Gate: масса, предвкушение, накопленная воля --------
        mass = torch.linalg.vector_norm(Z, dim=(-2, -1))
        anticipation = torch.linalg.vector_norm(Z_gluon, dim=(-2, -1))

        mass_src, mass_dst = mass[src], mass[dst]
        anticipation_src = anticipation[src]

        will_energy = F.softplus(
            mass_dst * anticipation_src + self.will_power - self.will_threshold
        )
        alpha = torch.sigmoid(will_energy).view(-1, 1, 1)

        # --- 3. Гравитация ----------------------------------------------
        gravity = mass_dst / (mass_src + mass_dst + self.eps)
        gravity_weight = gravity.view(-1, 1, 1)

        # --- 4. Нормализация рёбер --------------------------------------
        norm = self.edge_normalization(edge_index, N, Z.real.dtype, Z.device)

        # --- 5. Глюонный канал ------------------------------------------
        Z_gluon_t = self.parallel_transport(Z_gluon[src], U)
        Z_gluon_d = Z_gluon[dst]
        resonance_gluon = self.compute_resonance(Z_gluon_t, Z_gluon_d)
        msg_gluon = Z_gluon_t * resonance_gluon.unsqueeze(-1) * norm.view(-1, 1, 1)

        # --- 6. Виртуальный фотон (мышление/интерпретация электрона) ---
        Z_photon_t = self.parallel_transport(Z_photon[src], U)
        Z_photon_d = Z_photon[dst]

        dst_amp = torch.linalg.vector_norm(Z_photon_d, dim=-1)
        phase = self.electron_phase_mlp(dst_amp)
        Z_photon_t = Z_photon_t * torch.exp(1j * phase).unsqueeze(-1)
resonance_photon = self.compute_resonance(Z_photon_t, Z_photon_d)
        msg_photon = Z_photon_t * resonance_photon.unsqueeze(-1) * norm.view(-1, 1, 1)

        # --- 7. Реальные фотоны (слова/звук) — чистый вход -------------
        Z_real_t = self.parallel_transport(Z_real[src], U)
        if edge_attr is not None:
            if self.real_photon_proj is None:
                raise ValueError(
                    "edge_attr передан в forward, но слой был создан без edge_attr_dim."
                )
            real_scalar = torch.sigmoid(self.real_photon_proj(edge_attr)).unsqueeze(-1)
            msg_real = Z_real_t * real_scalar * norm.view(-1, 1, 1)
        else:
            msg_real = Z_real_t * norm.view(-1, 1, 1)

        # --- 8. Смешивание ----------------------------------------------
        msg = (alpha * msg_gluon + (1.0 - alpha) * msg_photon + msg_real) * gravity_weight

        # --- 9. Эхо -----------------------------------------------------
        alpha_2d = alpha.squeeze(-1)
        resonance = alpha_2d * resonance_gluon + (1.0 - alpha_2d) * resonance_photon

        if prev_resonance is not None:
            if prev_resonance.shape[-1] != self.out_channels and self.W_echo is not None:
                echo_proj = torch.einsum("ef,fg->eg", prev_resonance, self.W_echo)
            else:
                echo_proj = prev_resonance

            decay = torch.sigmoid(self.vibration_decay_logit)
            echo = decay * echo_proj + (1.0 - decay) * resonance
        else:
            echo = resonance

        msg = msg * echo.unsqueeze(-1)
        msg = self.complex_dropout(msg)

        # --- 10. Агрегация ----------------------------------------------
        aggregated = torch.zeros(N, self.out_channels, 3, dtype=Z.dtype, device=Z.device)
        aggregated.index_add_(0, dst, msg)

        # --- 11. Self + Residual ----------------------------------------
        out = Z_self + aggregated
        if self.residual:
            if self.W_residual is None:
                out = out + Z
            else:
                out = out + self.project_features(Z, self.W_residual)

        # --- 12. Активация и нормализация -------------------------------
        out = self.gauge_activation(out)
        mag = torch.linalg.vector_norm(out, dim=-1)
        mag = self.norm(mag.real)
        mag = F.softplus(mag) + 1e-3
        scale = mag / (torch.linalg.vector_norm(out, dim=-1) + self.eps)
        out = out * scale.unsqueeze(-1)

        # --- 13. Диагностика --------------------------------------------
        diagnostics = None
        if return_diagnostics:
            diagnostics = ResonanceDiagnostics(
                mean_resonance_gluon=resonance_gluon.mean().item(),
                mean_resonance_photon=resonance_photon.mean().item(),
                mean_alpha=alpha.mean().item(),
                mean_gravity=gravity.mean().item(),
                mean_will_energy=will_energy.mean().item(),
                will_power=self.will_power.item(),
                mean_state_norm=torch.linalg.vector_norm(out, dim=-1).mean().item(),
                mean_message_norm=torch.linalg.vector_norm(msg, dim=-1).mean().item(),
            )

        return out, resonance, diagnostics


# =============================================================================
# Wilson loop & Blocks (Без изменений)
# =============================================================================

def wilson_loop_curvature_loss(
    U_ij: torch.Tensor,
    U_jk: torch.Tensor,
    U_ki: torch.Tensor,
) -> torch.Tensor:
    W = U_ij @ U_jk @ U_ki
    I = torch.eye(3, dtype=W.dtype, device=W.device)
    return torch.linalg.matrix_norm(W - I).pow(2).mean()


@dataclass
class GaugeDiagnostics:
    unitarity_error: float
    determinant_error: float
class SU3GaugeBlock(nn.Module):
    def init(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: Optional[int] = None,
        *,
        residual: bool = True,
        resonance_power: float = 1.0,
        dropout: float = 0.0,
        use_curvature_reg: bool = False,
    ):
        super().init()
        self.use_curvature_reg = use_curvature_reg
        self.gauge = SU3GaugeField(edge_attr_dim=edge_attr_dim)
        self.layer = SU3ResonanceLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_attr_dim,
            residual=residual,
            resonance_power=resonance_power,
            dropout=dropout,
        )

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        prev_resonance: Optional[torch.Tensor] = None,
        triangles: Optional[torch.Tensor] = None,
        *,
        return_diagnostics: bool = False,
    ):
        U = self.gauge.transport_matrix(edge_index, edge_attr)
        Z_out, resonance, diagnostics = self.layer(
            Z,
            edge_index,
            U,
            edge_attr=edge_attr,
            prev_resonance=prev_resonance,
            return_diagnostics=return_diagnostics,
        )

        curvature_loss = None
        if self.use_curvature_reg and triangles is not None and triangles.numel() > 0:
            U_ij, U_jk, U_ki = U[triangles[0]], U[triangles[1]], U[triangles[2]]
            curvature_loss = wilson_loop_curvature_loss(U_ij, U_jk, U_ki)

        return Z_out, U, resonance, diagnostics, curvature_loss

    def gauge_diagnostics(self, U: torch.Tensor) -> GaugeDiagnostics:
        return GaugeDiagnostics(
            unitarity_error=SU3GaugeField.unitarity_error(U).item(),
            determinant_error=SU3GaugeField.determinant_error(U).item(),
        )


class SU3ResonanceGNN(nn.Module):
    def init(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_attr_dim: Optional[int] = None,
        *,
        resonance_power: float = 1.0,
        dropout: float = 0.0,
        use_curvature_reg: bool = False,
    ):
        super().init()
        self.num_layers = num_layers
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        self.blocks = nn.ModuleList([
            SU3GaugeBlock(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                edge_attr_dim=edge_attr_dim,
                residual=True,
                resonance_power=resonance_power,
                dropout=dropout,
                use_curvature_reg=use_curvature_reg,
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        triangles: Optional[torch.Tensor] = None,
    ):
        all_diag: List[GaugeDiagnostics] = []
        total_curvature: Optional[torch.Tensor] = None
        prev_resonance: Optional[torch.Tensor] = None

        for block in self.blocks:
            Z, U, resonance, _, c_loss = block(
                Z, edge_index, edge_attr=edge_attr,
                prev_resonance=prev_resonance, triangles=triangles,
            )
            prev_resonance = resonance
            all_diag.append(block.gauge_diagnostics(U))
            if c_loss is not None:
                total_curvature = c_loss if total_curvature is None else total_curvature + c_loss

        return Z, all_diag, total_curvature, prev_resonance


# =============================================================================
# Tests
# =============================================================================
@torch.no_grad()
def random_su3(num_nodes: int, scale: float = 0.1, device=None) -> torch.Tensor:
    T = get_gell_mann_matrices(device=device)
    phi = torch.randn(num_nodes, 8, device=device) * scale
    A = 1j * torch.einsum("na,abc->nbc", phi.to(T.dtype), T)
    return SU3GaugeField.exponential_map(A)


@torch.no_grad()
def gauge_covariance_test(
    layer: SU3ResonanceLayer,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    U: torch.Tensor,
    G: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
) -> float:
    src, dst = edge_index
    Z_g = torch.einsum("nab,nfb->nfa", G, Z)
    U_g = torch.einsum(
        "eij,ejk,ekl->eil",
        G[dst], U, G[src].conj().transpose(-2, -1),
    )
    out, _, _ = layer(Z, edge_index, U, edge_attr=edge_attr)
    out_g, _, _ = layer(Z_g, edge_index, U_g, edge_attr=edge_attr)
    expected = torch.einsum("nab,nfb->nfa", G, out)
    num = torch.linalg.vector_norm(out_g - expected)
    den = torch.linalg.vector_norm(expected) + 1e-8
    return (num / den).item()


def run_self_test() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, E = 32, 96
    F_in, F_hidden, F_out = 8, 16, 4
    num_layers = 3
    edge_attr_dim = 5

    edge_index = torch.randint(0, N, (2, E), device=device)
    edge_index = make_undirected(edge_index)
    E = edge_index.size(1)

    edge_attr = torch.randn(E, edge_attr_dim, device=device)
    triangles = torch.stack([
        torch.randint(0, E, (12,), device=device),
        torch.randint(0, E, (12,), device=device),
        torch.randint(0, E, (12,), device=device),
    ], dim=0)

    Z = torch.randn(N, F_in, 3, device=device, dtype=torch.complex64)

    print("=" * 70)
    print("SU(3) Gauge-Covariant Resonance Layer — v3.4")
    print("Gluon = tactile | Real photon = words | Virtual photon = thinking")
    print("=" * 70)

    block = SU3GaugeBlock(
        in_channels=F_in,
        out_channels=F_hidden,
        edge_attr_dim=edge_attr_dim,
        residual=True,
        use_curvature_reg=True,
    ).to(device)

    Z_out, U, resonance, diag, c_loss = block(
        Z, edge_index, edge_attr=edge_attr, triangles=triangles, return_diagnostics=True
    )
    g_diag = block.gauge_diagnostics(U)

    print(f"device:              {device}")
    print(f"input shape:         {tuple(Z.shape)}")
    print(f"output shape:        {tuple(Z_out.shape)}")
    print(f"edges:               {E}")
    print()
    print(f"unitarity error:     {g_diag.unitarity_error:.3e}")
    print(f"determinant error:   {g_diag.determinant_error:.3e}")
    if diag is not None:
        print(f"mean alpha (will):   {diag.mean_alpha:.4f}")
        print(f"mean will energy:    {diag.mean_will_energy:.4f}")
        print(f"will_power (accum):  {diag.will_power:.4f}")
        print(f"mean gravity:        {diag.mean_gravity:.4f}")
        print(f"resonance gluon:     {diag.mean_resonance_gluon:.4f}")
        print(f"resonance photon:    {diag.mean_resonance_photon:.4f}")
    if c_loss is not None:
        print(f"curvature loss:      {c_loss.item():.3e}")

    G = random_su3(N, device=device)
    cov_err = gauge_covariance_test(
        block.layer, Z, edge_index, U, G, edge_attr=edge_attr
    )
    print(f"equivariance error:  {cov_err:.3e}")

    with torch.no_grad():
        big_attr = torch.randn(E, edge_attr_dim, device=device) * 10.0
        U_big = block.gauge.transport_matrix(edge_index, edge_attr=big_attr)
        print(f"large-phi unitary:   {SU3GaugeField.unitarity_error(U_big):.3e}")
        print(f"large-phi det-1:     {SU3GaugeField.determinant_error(U_big):.3e}")

    print("\nMulti-layer GNN test:")
    gnn = SU3ResonanceGNN(
        in_channels=F_in,
        hidden_channels=F_hidden,
        out_channels=F_out,
        num_layers=num_layers,
        edge_attr_dim=edge_attr_dim,
        dropout=0.1,
    ).to(device)
Z_final, all_diag, total_c, final_resonance = gnn(
        Z, edge_index, edge_attr=edge_attr, triangles=triangles
    )
    print(f"  final shape:       {tuple(Z_final.shape)}")
    for i, d in enumerate(all_diag):
        print(f"  layer {i} unitary:  {d.unitarity_error:.3e}")

    print("\nWill power per layer (accumulated overcoming):")
    for i, block in enumerate(gnn.blocks):
        print(f"  layer {i}: {block.layer.will_power.item():.4f}")

    print("\nPASS CONDITIONS")
    print(f"  unitary:           {g_diag.unitarity_error < 1e-4}")
    print(f"  determinant:       {g_diag.determinant_error < 1e-4}")
    print(f"  gauge covariance:  {cov_err < 1e-4}")
    print("=" * 70)


if name == "main":
    run_self_test()
