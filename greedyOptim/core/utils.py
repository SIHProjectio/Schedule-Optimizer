import numpy as np
from typing import Dict, List, Optional, Tuple


CERTIFICATE_STATUS_MAP = {
    'PENDING': 'Expiring-Soon',
    'IN_PROGRESS': 'Expiring-Soon',
    'ISSUED': 'Valid',
    'EXPIRED': 'Expired',
    'SUSPENDED': 'Suspended',
    'REVOKED': 'Expired',
    'RENEWED': 'Valid',
    'CANCELLED': 'Expired',
}

COMPONENT_STATUS_MAP = {
    'EXCELLENT': 'Good',
    'GOOD': 'Good',
    'FAIR': 'Fair',
    'POOR': 'Warning',
    'CRITICAL': 'Critical',
    'FAILED': 'Critical',
}

OPERATIONAL_STATUS_MAP = {
    'IN_SERVICE': 'In-Service',
    'STANDBY': 'Standby',
    'MAINTENANCE': 'Maintenance',
    'OUT_OF_SERVICE': 'Out-of-Order',
    'TESTING': 'Maintenance',
}


def normalize_certificate_status(status: str) -> str:
    """Normalize certificate status to internal format."""
    return CERTIFICATE_STATUS_MAP.get(status, status)


def normalize_component_status(status: str) -> str:
    """Normalize component status to internal format."""
    return COMPONENT_STATUS_MAP.get(status, status)


def normalize_operational_status(status: str) -> str:
    """Normalize operational status to internal format."""
    return OPERATIONAL_STATUS_MAP.get(status, status)


def decode_solution(x: np.ndarray) -> np.ndarray:
    """Decode continuous values to discrete actions (0=Service, 1=Standby, 2=Maintenance)."""
    return np.clip(np.round(x), 0, 2).astype(int)


def create_block_assignment(
    trainset_solution: np.ndarray,
    num_blocks: int,
    randomize: bool = False
) -> np.ndarray:
    """Create block assignments for a trainset solution.
    
    Args:
        trainset_solution: Array where 0=Service, 1=Standby, 2=Maintenance
        num_blocks: Total number of service blocks
        randomize: Whether to add randomization to block assignments
        
    Returns:
        Array mapping each block index to a trainset index (-1 if unassigned)
    """
    service_indices = np.where(trainset_solution == 0)[0]
    
    if len(service_indices) == 0:
        return np.full(num_blocks, -1, dtype=int)
    
    block_sol = np.zeros(num_blocks, dtype=int)
    for i in range(num_blocks):
        block_sol[i] = service_indices[i % len(service_indices)]
    
    if randomize:
        np.random.shuffle(block_sol)
        for i in range(len(block_sol)):
            if block_sol[i] not in service_indices:
                block_sol[i] = np.random.choice(service_indices)
    
    return block_sol


def repair_block_assignment(
    block_solution: np.ndarray,
    trainset_solution: np.ndarray
) -> np.ndarray:
    """Repair block assignments to only assign to service trains.
    
    Args:
        block_solution: Current block assignments
        trainset_solution: Current trainset assignments
        
    Returns:
        Repaired block assignments
    """
    repaired = block_solution.copy()
    service_indices = np.where(trainset_solution == 0)[0]
    
    if len(service_indices) == 0:
        return np.full(len(block_solution), -1, dtype=int)
    
    for i in range(len(repaired)):
        if repaired[i] not in service_indices:
            repaired[i] = np.random.choice(service_indices)
    
    return repaired


def mutate_block_assignment(
    block_solution: np.ndarray,
    service_indices: np.ndarray,
    mutation_rate: float = 0.1
) -> np.ndarray:
    """Mutate block assignment with given rate.
    
    Args:
        block_solution: Current block assignments
        service_indices: Indices of trainsets in service
        mutation_rate: Probability of mutating each block
        
    Returns:
        Mutated block assignments
    """
    mutated = block_solution.copy()
    
    if len(service_indices) == 0:
        return mutated
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.choice(service_indices)
    
    return mutated


def extract_solution_groups(
    solution: np.ndarray,
    trainset_ids: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """Extract service, standby, and maintenance trainsets from solution.
    
    Args:
        solution: Array of trainset assignments (0=Service, 1=Standby, 2=Maintenance)
        trainset_ids: List of trainset IDs
        
    Returns:
        Tuple of (service_trainsets, standby_trainsets, maintenance_trainsets)
    """
    service = [trainset_ids[i] for i, v in enumerate(solution) if v == 0]
    standby = [trainset_ids[i] for i, v in enumerate(solution) if v == 1]
    maintenance = [trainset_ids[i] for i, v in enumerate(solution) if v == 2]
    return service, standby, maintenance


def build_block_assignments_dict(
    block_solution: np.ndarray,
    service_trainsets: List[str],
    trainset_ids: List[str],
    all_blocks: List[Dict]
) -> Dict[str, List[str]]:
    """Build block assignments dictionary from block solution.
    
    Args:
        block_solution: Array mapping block index to trainset index
        service_trainsets: List of trainset IDs in service
        trainset_ids: Complete list of trainset IDs
        all_blocks: List of all service block dictionaries
        
    Returns:
        Dictionary mapping trainset_id -> list of block_ids
    """
    block_assignments: Dict[str, List[str]] = {}
    
    if block_solution is None:
        return block_assignments
    
    for ts_id in service_trainsets:
        block_assignments[ts_id] = []
    
    for block_idx, train_idx in enumerate(block_solution):
        if 0 <= train_idx < len(trainset_ids):
            ts_id = trainset_ids[int(train_idx)]
            if ts_id in block_assignments and block_idx < len(all_blocks):
                block_id = all_blocks[block_idx]['block_id']
                block_assignments[ts_id].append(block_id)
    
    return block_assignments
