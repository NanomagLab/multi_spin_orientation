from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import dist, log10
from time import perf_counter
from typing import Iterable, Optional
import numpy as np  

from lib.spin_utils import get_composition_rule

# UNIT_LENGTH = np.sqrt(3.)
UNIT_LENGTH = 2. / np.sqrt(3.)

# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class Particle:
    """
    A particle in either prev or post snapshot.

    Attributes
    ----------
    id:
        Global unique id within its side.
    type:
        Particle type. Vacuum 0 is not stored as a Particle.
    y, x:
        Position on the plane.
    side:
        "prev" or "post"
    """
    id: int
    type: int
    y: float
    x: float
    side: str

    @property
    def pos(self) -> tuple[float, float]:
        return (self.y, self.x)

    def distance_to(self, other: "Particle") -> float:
        return dist(self.pos, other.pos)


@dataclass
class Event:
    """
    Candidate event.

    Only a small amount of information is strictly necessary for solving:
    - prev_ids
    - post_ids
    - cost

    kind and rule are kept mainly for debugging / interpretation.
    """
    event_id: int
    kind: str
    prev_ids: list[int] = field(default_factory=list)
    post_ids: list[int] = field(default_factory=list)
    cost: float = 0.0
    rule: Optional[tuple[int, ...]] = None


@dataclass
class Trajectory:
    """One logical particle path: (iteration, y, x) samples in order."""

    type: int
    traj: list[tuple[int, float, float]] = field(default_factory=list)


# =========================
# Normalization helpers
# =========================


def build_particles_from_typed_arrays(
    typed_particles: list,
    side: str,
) -> list[Particle]:
    """
    Convert typed particle arrays into a flat particle list.

    Parameters
    ----------
    typed_particles:
        Length N-1 list.
        typed_particles[i] is an (n_i, 2) array-like of (y, x) for particle type i+1.
    side:
        "prev" or "post"
    """
    particles: list[Particle] = []
    pid = 0
    for type_idx, arr in enumerate(typed_particles, start=1):
        for y, x in arr:
            particles.append(Particle(id=pid, type=type_idx, y=float(y), x=float(x), side=side))
            pid += 1
    return particles


# =========================
# Small utility helpers
# =========================


def _canon_ids(ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(ids))



def _event_key(kind: str, prev_ids: Iterable[int], post_ids: Iterable[int], rule=None):
    if rule is None:
        rule_key = None
    else:
        rule_key = tuple(rule)
    return (kind, _canon_ids(prev_ids), _canon_ids(post_ids), rule_key)



def _pair_type_matches(t1: int, t2: int, a: int, b: int) -> bool:
    return tuple(sorted((t1, t2))) == tuple(sorted((a, b)))


EPSILON = 1e-6



def _leq_with_eps(lhs: float, rhs: float, eps: float = EPSILON) -> bool:
    return lhs <= rhs + eps



def _close_with_eps(a: float, b: float, eps: float = EPSILON) -> bool:
    return abs(a - b) <= eps

# =========================
# Cost functions
# =========================


def _identity_cost(prev_p: Particle, post_p: Particle) -> float:
    return prev_p.distance_to(post_p) ** 2



def _decomposition_cost(prev_p: Particle, post_p1: Particle, post_p2: Particle) -> float:
    return prev_p.distance_to(post_p1) ** 2 + prev_p.distance_to(post_p2) ** 2



def _creation_cost(post_p1: Particle, post_p2: Particle) -> float:
    pair_term = (post_p1.distance_to(post_p2) / 2.0) ** 2
    return pair_term



def _composition_cost(prev_p1: Particle, prev_p2: Particle, post_p: Particle) -> float:
    return prev_p1.distance_to(post_p) ** 2 + prev_p2.distance_to(post_p) ** 2



def _annihilation_cost(prev_p1: Particle, prev_p2: Particle) -> float:
    pair_term = (prev_p1.distance_to(prev_p2) / 2.0) ** 2
    return pair_term


# =========================
# Event builders
# =========================


def _add_event_if_new(
    events: list[Event],
    seen_keys: set,
    kind: str,
    prev_ids: list[int],
    post_ids: list[int],
    cost: float,
    rule: Optional[tuple[int, ...]] = None,
) -> None:
    key = _event_key(kind, prev_ids, post_ids, rule)
    if key in seen_keys:
        return
    seen_keys.add(key)
    events.append(
        Event(
            event_id=len(events),
            kind=kind,
            prev_ids=list(prev_ids),
            post_ids=list(post_ids),
            cost=float(cost),
            rule=rule,
        )
    )


def _build_identity_events(
    prev_particles: list[Particle],
    post_particles: list[Particle],
    max_distance: float = UNIT_LENGTH,
) -> list[Event]:
    events: list[Event] = []
    seen_keys: set = set()

    for prev_p in prev_particles:
        for post_p in post_particles:
            if prev_p.type != post_p.type:
                continue
            if not _leq_with_eps(prev_p.distance_to(post_p), max_distance):
                continue
            _add_event_if_new(
                events,
                seen_keys,
                kind="identity",
                prev_ids=[prev_p.id],
                post_ids=[post_p.id],
                cost=_identity_cost(prev_p, post_p),
                rule=None,
            )
    return events



def _build_decomposition_events(
    prev_particles: list[Particle],
    post_particles: list[Particle],
    decomposition_rule: list,
    max_distance: float = UNIT_LENGTH,
) -> list[Event]:
    events: list[Event] = []
    seen_keys: set = set()

    for prev_p in prev_particles:
        allowed_pairs = decomposition_rule[prev_p.type]
        for a, b in allowed_pairs:
            for post_p1, post_p2 in combinations(post_particles, 2):
                if not _pair_type_matches(post_p1.type, post_p2.type, int(a), int(b)):
                    continue
                if not _leq_with_eps(prev_p.distance_to(post_p1), max_distance):
                    continue
                if not _leq_with_eps(prev_p.distance_to(post_p2), max_distance):
                    continue
                _add_event_if_new(
                    events,
                    seen_keys,
                    kind="decomposition",
                    prev_ids=[prev_p.id],
                    post_ids=[post_p1.id, post_p2.id],
                    cost=_decomposition_cost(prev_p, post_p1, post_p2),
                    rule=(int(a), int(b)),
                )
    return events



def _build_creation_events(
    post_particles: list[Particle],
    decomposition_rule: list,
    max_distance: float = UNIT_LENGTH * 2.0,
) -> list[Event]:
    events: list[Event] = []
    seen_keys: set = set()

    creation_pairs = decomposition_rule[0]
    for a, b in creation_pairs:
        for post_p1, post_p2 in combinations(post_particles, 2):
            if not _pair_type_matches(post_p1.type, post_p2.type, int(a), int(b)):
                continue
            if not _leq_with_eps(post_p1.distance_to(post_p2), max_distance):
                continue
            _add_event_if_new(
                events,
                seen_keys,
                kind="creation",
                prev_ids=[],
                post_ids=[post_p1.id, post_p2.id],
                cost=_creation_cost(post_p1, post_p2),
                rule=(int(a), int(b)),
            )
    return events



def _build_composition_events(
    prev_particles: list[Particle],
    post_particles: list[Particle],
    decomposition_rule: list,
    max_distance: float = UNIT_LENGTH,
) -> list[Event]:
    events: list[Event] = []
    seen_keys: set = set()

    for post_p in post_particles:
        allowed_pairs = decomposition_rule[post_p.type]
        for a, b in allowed_pairs:
            for prev_p1, prev_p2 in combinations(prev_particles, 2):
                if not _pair_type_matches(prev_p1.type, prev_p2.type, int(a), int(b)):
                    continue
                if not _leq_with_eps(prev_p1.distance_to(post_p), max_distance):
                    continue
                if not _leq_with_eps(prev_p2.distance_to(post_p), max_distance):
                    continue
                _add_event_if_new(
                    events,
                    seen_keys,
                    kind="composition",
                    prev_ids=[prev_p1.id, prev_p2.id],
                    post_ids=[post_p.id],
                    cost=_composition_cost(prev_p1, prev_p2, post_p),
                    rule=(int(a), int(b)),
                )
    return events



def _build_annihilation_events(
    prev_particles: list[Particle],
    decomposition_rule: list,
    max_distance: float = UNIT_LENGTH * 2.0,
) -> list[Event]:
    events: list[Event] = []
    seen_keys: set = set()

    annihilation_pairs = decomposition_rule[0]
    for a, b in annihilation_pairs:
        for prev_p1, prev_p2 in combinations(prev_particles, 2):
            if not _pair_type_matches(prev_p1.type, prev_p2.type, int(a), int(b)):
                continue
            if not _leq_with_eps(prev_p1.distance_to(prev_p2), max_distance):
                continue
            _add_event_if_new(
                events,
                seen_keys,
                kind="annihilation",
                prev_ids=[prev_p1.id, prev_p2.id],
                post_ids=[],
                cost=_annihilation_cost(prev_p1, prev_p2),
                rule=(int(a), int(b)),
            )
    return events



def build_events(
    prev_particles: list[Particle],
    post_particles: list[Particle],
    decomposition_rule: list,
) -> list[Event]:
    """
    Build candidate events.

    Supported event types:
    - identity      : 1 -> 1
    - decomposition : 1 -> 2
    - creation      : 0 -> 2
    - composition   : 2 -> 1
    - annihilation  : 2 -> 0

    Composition / annihilation are interpreted as reverse processes of
    decomposition / creation using the same rule table.
    """
    events: list[Event] = []
    events.extend(_build_identity_events(prev_particles, post_particles))
    events.extend(_build_decomposition_events(prev_particles, post_particles, decomposition_rule))
    events.extend(_build_creation_events(post_particles, decomposition_rule))
    events.extend(_build_composition_events(prev_particles, post_particles, decomposition_rule))
    events.extend(_build_annihilation_events(prev_particles, decomposition_rule))

    # Re-assign event ids after concatenation for consistency.
    for eid, event in enumerate(events):
        event.event_id = eid
    return events


# =========================
# Optional preprocessing
# =========================


def peel_off_trivial_identities(
    prev_particles: list[Particle],
    post_particles: list[Particle],
    tol: float = 1e-12,
) -> tuple[list[Event], list[Particle], list[Particle]]:
    """
    Greedily remove exact same-type, same-position pairs.

    This is the preprocessing idea discussed earlier:
    handle obvious unchanged particles first, then solve the residual problem.
    """
    fixed_events: list[Event] = []
    used_prev: set[int] = set()
    used_post: set[int] = set()

    for prev_p in prev_particles:
        for post_p in post_particles:
            if post_p.id in used_post:
                continue
            if prev_p.type != post_p.type:
                continue
            if not _close_with_eps(prev_p.y, post_p.y, tol) or not _close_with_eps(prev_p.x, post_p.x, tol):
                continue

            fixed_events.append(
                Event(
                    event_id=len(fixed_events),
                    kind="identity",
                    prev_ids=[prev_p.id],
                    post_ids=[post_p.id],
                    cost=0.0,
                    rule=None,
                )
            )
            used_prev.add(prev_p.id)
            used_post.add(post_p.id)
            break

    prev_rest = [p for p in prev_particles if p.id not in used_prev]
    post_rest = [p for p in post_particles if p.id not in used_post]
    return fixed_events, prev_rest, post_rest


# =========================
# DFS solver
# =========================


def _build_events_by_prev(events: list[Event], prev_ids: list[int]) -> dict[int, list[Event]]:
    events_by_prev = {pid: [] for pid in prev_ids}
    for event in events:
        for pid in event.prev_ids:
            if pid in events_by_prev:
                events_by_prev[pid].append(event)
    return events_by_prev


def _build_events_by_post(events: list[Event], post_ids: list[int]) -> dict[int, list[Event]]:
    events_by_post = {qid: [] for qid in post_ids}
    for event in events:
        for qid in event.post_ids:
            if qid in events_by_post:
                events_by_post[qid].append(event)
    return events_by_post


def prune_particles_without_events(
    events: list[Event],
    prev_rest: list[Particle],
    post_rest: list[Particle],
) -> tuple[list[Event], list[Particle], list[Particle], list[int], list[int]]:
    """
    Repeatedly remove nodes (prev/post ids) that have zero incident events in the
    current candidate list, and drop any event touching a removed node, until fixed point.
    Discarded ids are those removed as having no incident events at the time of removal.
    """
    active_prev = {p.id for p in prev_rest}
    active_post = {p.id for p in post_rest}
    events_work = list(events)
    discarded_prev: list[int] = []
    discarded_post: list[int] = []

    while True:
        prev_ids = sorted(active_prev)
        post_ids = sorted(active_post)
        by_prev = _build_events_by_prev(events_work, prev_ids)
        by_post = _build_events_by_post(events_work, post_ids)
        iso_prev = {p for p in active_prev if not by_prev.get(p)}
        iso_post = {q for q in active_post if not by_post.get(q)}
        if not iso_prev and not iso_post:
            break
        discarded_prev.extend(sorted(iso_prev))
        discarded_post.extend(sorted(iso_post))
        active_prev -= iso_prev
        active_post -= iso_post

        def event_still_valid(e: Event) -> bool:
            return all(pid in active_prev for pid in e.prev_ids) and all(
                qid in active_post for qid in e.post_ids
            )

        events_work = [e for e in events_work if event_still_valid(e)]

    prev_f = sorted((p for p in prev_rest if p.id in active_prev), key=lambda p: p.id)
    post_f = sorted((p for p in post_rest if p.id in active_post), key=lambda p: p.id)
    return events_work, prev_f, post_f, discarded_prev, discarded_post

def _is_compatible(event: Event, used_prev: set[int], used_post: set[int]) -> bool:
    for pid in event.prev_ids:
        if pid in used_prev:
            return False
    for qid in event.post_ids:
        if qid in used_post:
            return False
    return True



def solve_events_by_dfs(
    events: list[Event],
    prev_ids: list[int],
    post_ids: list[int],
) -> tuple[list[Event], bool, int, int]:
    """
    Solve by DFS / backtracking.

    Returns (selected_events, is_exact_cover, unsolved_prev_n, unsolved_post_n).
    unsolved_* are nonzero only when is_exact_cover is False (uncovered endpoints
    on the active subgraph). Logging of prune + unsolved is done in func.
    """
    events_by_post = _build_events_by_post(events, post_ids)
    events_by_prev = _build_events_by_prev(events, prev_ids)
    start_t = perf_counter()
    progress_every = 200_000
    max_nodes_visited = 1_000_000
    nodes_visited = 0
    next_progress_log = progress_every
    hit_visit_cap = False

    best_cost = float("inf")
    best_solution: list[Event] | None = None

    total_endpoints = len(prev_ids) + len(post_ids)
    best_partial_cov = -1
    best_partial_cost = float("inf")
    best_partial_solution: list[Event] | None = None
    prev_degrees = [len(events_by_prev.get(pid, [])) for pid in prev_ids]
    post_degrees = [len(events_by_post.get(qid, [])) for qid in post_ids]
    avg_degree = (
        (sum(prev_degrees) + sum(post_degrees)) / max(total_endpoints, 1)
        if total_endpoints > 0
        else 0.0
    )
    rough_log10_states = total_endpoints * log10(max(avg_degree, 1.0))
    print(
        "dfs_estimate "
        f"prev_degree[min/avg/max]={min(prev_degrees, default=0)}/{(sum(prev_degrees)/max(len(prev_degrees),1)):.2f}/{max(prev_degrees, default=0)} "
        f"post_degree[min/avg/max]={min(post_degrees, default=0)}/{(sum(post_degrees)/max(len(post_degrees),1)):.2f}/{max(post_degrees, default=0)} "
        f"rough_log10_states~{rough_log10_states:.1f}"
    )

    def choose_next_uncovered(used_prev: set[int], used_post: set[int]) -> tuple[str, int]:
        remaining_prev = [pid for pid in prev_ids if pid not in used_prev]
        if remaining_prev:
            best_pid = min(remaining_prev, key=lambda pid: len(events_by_prev.get(pid, [])))
            return ("prev", best_pid)

        remaining_post = [qid for qid in post_ids if qid not in used_post]
        if remaining_post:
            best_qid = min(remaining_post, key=lambda qid: len(events_by_post.get(qid, [])))
            return ("post", best_qid)

        raise ValueError("No remaining particle to choose from.")

    def dfs(
        used_prev: set[int],
        used_post: set[int],
        selected: list[Event],
        cur_cost: float,
    ) -> None:
        nonlocal best_cost, best_solution, best_partial_cov, best_partial_cost, best_partial_solution
        nonlocal nodes_visited, next_progress_log, hit_visit_cap
        if hit_visit_cap:
            return
        nodes_visited += 1
        if nodes_visited >= max_nodes_visited:
            hit_visit_cap = True
            print(f"dfs_cap_reached visited={nodes_visited} (cap={max_nodes_visited})")
            return
        if nodes_visited >= next_progress_log:
            elapsed = perf_counter() - start_t
            rate = nodes_visited / max(elapsed, 1e-9)
            print(
                "dfs_progress "
                f"visited={nodes_visited} elapsed_s={elapsed:.1f} rate={rate:.0f}/s "
                f"best_partial_cov={best_partial_cov}/{total_endpoints} "
                f"best_cost={'inf' if best_cost == float('inf') else f'{best_cost:.4f}'}"
            )
            next_progress_log += progress_every

        if len(used_prev) == len(prev_ids) and len(used_post) == len(post_ids):
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_solution = selected.copy()
            return

        cov = len(used_prev) + len(used_post)
        if cov > best_partial_cov or (cov == best_partial_cov and cur_cost < best_partial_cost):
            best_partial_cov = cov
            best_partial_cost = cur_cost
            best_partial_solution = selected.copy()

        side, target_id = choose_next_uncovered(used_prev, used_post)
        if side == "post":
            candidate_events = events_by_post.get(target_id, [])
        else:
            candidate_events = events_by_prev.get(target_id, [])

        for event in candidate_events:
            if not _is_compatible(event, used_prev, used_post):
                continue

            new_cost = cur_cost + event.cost
            if new_cost >= best_cost:
                continue

            selected.append(event)

            added_prev: list[int] = []
            added_post: list[int] = []

            for pid in event.prev_ids:
                if pid not in used_prev:
                    used_prev.add(pid)
                    added_prev.append(pid)

            for post_id in event.post_ids:
                if post_id not in used_post:
                    used_post.add(post_id)
                    added_post.append(post_id)

            dfs(used_prev, used_post, selected, new_cost)

            for pid in added_prev:
                used_prev.remove(pid)
            for post_id in added_post:
                used_post.remove(post_id)
            selected.pop()

    dfs(used_prev=set(), used_post=set(), selected=[], cur_cost=0.0)

    if best_solution is not None:
        elapsed = perf_counter() - start_t
        print(
            "dfs_done "
            f"exact_cover=True visited={nodes_visited} elapsed_s={elapsed:.1f}"
        )
        return best_solution, True, 0, 0

    out = best_partial_solution if best_partial_solution is not None else []
    covered_prev: set[int] = set()
    covered_post: set[int] = set()
    for e in out:
        covered_prev.update(e.prev_ids)
        covered_post.update(e.post_ids)
    unsolved_prev = len(prev_ids) - len(covered_prev)
    unsolved_post = len(post_ids) - len(covered_post)
    elapsed = perf_counter() - start_t
    print(
        "dfs_done "
        f"exact_cover=False visited={nodes_visited} elapsed_s={elapsed:.1f} "
        f"unsolved_prev={unsolved_prev} unsolved_post={unsolved_post}"
    )
    return out, False, unsolved_prev, unsolved_post


# =========================
# Main entry
# =========================


def func(prev_particles: list, post_particles: list, decomposition_rule: list) -> list[Event]:
    """
    Parameters
    ----------
    prev_particles, post_particles:
        N-1 arrays, where N is the number of particle types including vacuum 0.
        The i-th array is (n_i, 2), referring to particle positions (y, x) of type i+1.

    decomposition_rule:
        List of N arrays.
        decomposition_rule[i] is an (m_i, 2) array of allowed decomposed particle types.
        decomposition_rule[0] corresponds to pair creation from vacuum.

    Returns
    -------
    list[Event]
        Selected event combination explaining all remaining particles.
    """
    prev_list = build_particles_from_typed_arrays(prev_particles, side="prev")
    post_list = build_particles_from_typed_arrays(post_particles, side="post")

    fixed_events, prev_rest, post_rest = peel_off_trivial_identities(prev_list, post_list)

    if not prev_rest and not post_rest:
        return fixed_events

    candidate_events = build_events(prev_rest, post_rest, decomposition_rule)
    candidate_events, prev_rest, post_rest, discarded_prev, discarded_post = prune_particles_without_events(
        candidate_events, prev_rest, post_rest
    )
    prev_ids = [p.id for p in prev_rest]
    post_ids = [p.id for p in post_rest]

    residual_solution, _exact, unsolved_prev_dfs, unsolved_post_dfs = solve_events_by_dfs(
        candidate_events, prev_ids, post_ids
    )

    # Re-assign event ids for the final combined list.
    final_events = fixed_events + residual_solution
    for eid, event in enumerate(final_events):
        event.event_id = eid
    return final_events


def _particle_midpoint_xy(p1: Particle, p2: Particle) -> tuple[float, float]:
    return ((p1.y + p2.y) * 0.5, (p1.x + p2.x) * 0.5)


def get_trajectory(
    defect_record: dict[int, np.ndarray],
    multi_spin_mode: str,
) -> list[Trajectory]:
    """
    Build per-particle trajectories from a time-ordered defect field record.

    Parameters
    ----------
    defect_record:
        ``{iteration: field}`` with each field shaped ``(..., 2)`` (same convention as
        ``field_to_particles_triangular``).
    composition_rule:
        Same as ``func`` / ``build_events`` decomposition table; ``len`` is the number
        of type slots including vacuum (index 0).

    Notes
    -----
    - Consecutive frames ``(k_i, k_{i+1})`` are matched with ``func``; events update
      trajectories as described (identity, creation, annihilation, decomposition, composition).
    - Annihilation / composition / decomposition end the incoming trajectories at merge/split
      rules; creation / decomposition / composition start new ``Trajectory`` objects as needed.
    - If ``func`` does not assign every particle (partial cover), prev-only particles have
      their trajectories **terminated** (no new sample at ``k_{i+1}``). Post-only particles get
      a **new** ``Trajectory`` starting at ``(k_{i+1}, y, x)``.
    """
    composition_rule = get_composition_rule(multi_spin_mode)

    n_types = len(composition_rule)
    iters = sorted(defect_record.keys())
    if len(iters) < 2:
        return []

    all_trajs: list[Trajectory] = []
    traj_by_post: dict[int, Trajectory] = {}

    for step_idx in range(len(iters) - 1):
        k0, k1 = iters[step_idx], iters[step_idx + 1]
        prev_field = defect_record[k0]
        post_field = defect_record[k1]
        print(f"trajectory_step prev_iter={k0}, post_iter={k1}", end="\r")

        prev_typed = field_to_particles(prev_field, n_types)
        post_typed = field_to_particles(post_field, n_types)
        prev_list = build_particles_from_typed_arrays(prev_typed, side="prev")
        post_list = build_particles_from_typed_arrays(post_typed, side="post")
        prev_by_id = {p.id: p for p in prev_list}
        post_by_id = {p.id: p for p in post_list}

        traj_by_prev: dict[int, Trajectory] = {}
        if step_idx == 0:
            for p in prev_list:
                t = Trajectory(type=p.type, traj=[(k0, p.y, p.x)])
                traj_by_prev[p.id] = t
                all_trajs.append(t)
        else:
            for p in prev_list:
                traj_by_prev[p.id] = traj_by_post[p.id]

        events = func(prev_typed, post_typed, composition_rule)
        traj_by_post = {}

        for ev in events:
            if ev.kind == "identity":
                pp = prev_by_id[ev.prev_ids[0]]
                pq = post_by_id[ev.post_ids[0]]
                t = traj_by_prev[pp.id]
                t.traj.append((k1, pq.y, pq.x))
                traj_by_post[pq.id] = t

            elif ev.kind == "creation":
                q1 = post_by_id[ev.post_ids[0]]
                q2 = post_by_id[ev.post_ids[1]]
                my, mx = _particle_midpoint_xy(q1, q2)
                t1 = Trajectory(
                    type=q1.type,
                    traj=[(k1, my, mx), (k1, q1.y, q1.x)],
                )
                t2 = Trajectory(
                    type=q2.type,
                    traj=[(k1, my, mx), (k1, q2.y, q2.x)],
                )
                all_trajs.extend([t1, t2])
                traj_by_post[q1.id] = t1
                traj_by_post[q2.id] = t2

            elif ev.kind == "annihilation":
                p1 = prev_by_id[ev.prev_ids[0]]
                p2 = prev_by_id[ev.prev_ids[1]]
                my, mx = _particle_midpoint_xy(p1, p2)
                traj_by_prev[p1.id].traj.append((k1, my, mx))
                traj_by_prev[p2.id].traj.append((k1, my, mx))

            elif ev.kind == "decomposition":
                p = prev_by_id[ev.prev_ids[0]]
                q1 = post_by_id[ev.post_ids[0]]
                q2 = post_by_id[ev.post_ids[1]]
                told = traj_by_prev[p.id]
                told.traj.append((k1, p.y, p.x))
                t1 = Trajectory(
                    type=q1.type,
                    traj=[(k1, p.y, p.x), (k1, q1.y, q1.x)],
                )
                t2 = Trajectory(
                    type=q2.type,
                    traj=[(k1, p.y, p.x), (k1, q2.y, q2.x)],
                )
                all_trajs.extend([t1, t2])
                traj_by_post[q1.id] = t1
                traj_by_post[q2.id] = t2

            elif ev.kind == "composition":
                p1 = prev_by_id[ev.prev_ids[0]]
                p2 = prev_by_id[ev.prev_ids[1]]
                q = post_by_id[ev.post_ids[0]]
                t1 = traj_by_prev[p1.id]
                t2 = traj_by_prev[p2.id]
                t1.traj.append((k1, q.y, q.x))
                t2.traj.append((k1, q.y, q.x))
                t_new = Trajectory(type=q.type, traj=[(k1, q.y, q.x)])
                all_trajs.append(t_new)
                traj_by_post[q.id] = t_new

            else:
                raise ValueError(f"unknown event kind: {ev.kind}")

        prev_used: set[int] = set()
        post_used: set[int] = set()
        for ev in events:
            prev_used.update(ev.prev_ids)
            post_used.update(ev.post_ids)
        prev_unmatched_nodes = len(prev_by_id) - len(prev_used)
        post_unmatched_nodes = len(post_by_id) - len(post_used)
        if prev_unmatched_nodes > 0 or post_unmatched_nodes > 0:
            print(
                f"prev_iter={k0}, post_iter={k1}, "
                f"prev_unmatched_nodes={prev_unmatched_nodes}, post_unmatched_nodes={post_unmatched_nodes}"
            )

        # Prev not in any event: trajectory ends here (no point at k1).
        # Post not assigned a trajectory yet: start a new trajectory at k1.
        # unmatched_prev_ids = set(prev_by_id.keys()) - prev_used  (trajectories end; no append)

        for q in post_list:
            if q.id not in traj_by_post:
                t_new = Trajectory(type=q.type, traj=[(k1, q.y, q.x)])
                all_trajs.append(t_new)
                traj_by_post[q.id] = t_new

    return all_trajs



def field_to_particles(field: np.ndarray, n_types: int) -> list[np.ndarray]:
    """
    Convert a field to particles for a triangular lattice.
    """
    assert field.ndim == 3 and field.shape[-1] == 2
    particles = []
    h, w = field.shape[:2]
    for type_ind in range(1, n_types):
        ii, jj, kk = np.where(field == type_ind)
        yy = (ii - h/2. + 0.5) * np.sqrt(3.) / 2. + (kk + 1.) * 0.5 / np.sqrt(3.)
        xx = (ii - h/2. + 0.5) * 0.5 + (jj - w/2. + 0.5) + (kk + 1.) * 0.5
        particles.append(np.stack([yy, xx], axis=-1))
    return particles


# def field_to_particles_hexagonal(field: np.ndarray, n_types: int) -> list[np.ndarray]:
#     """
#     Convert a field to particles for a hexagonal lattice.
#     """
#     assert field.ndim == 2
#     particles = []
#     h, w = field.shape
#     for type_ind in range(1, n_types):
#         ii, jj = np.where(field == type_ind)
#         yy = (ii - h/2. + 0.5) * np.sqrt(3.) / 2.
#         xx = (ii - h/2. + 0.5) * 0.5 + (jj - w/2. + 0.5)
#         particles.append(np.stack([yy, xx], axis=-1))
#     return particles