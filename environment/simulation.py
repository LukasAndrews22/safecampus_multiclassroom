import numpy as np

SEED = 42
np.random.seed(SEED)

def simulate_infections_n_classrooms(n_classes, alpha_m, beta, delta, current_infected, allowed_students, community_risk):
    # 1. Convert lists to arrays internally (this is the only new part)
    # We use 'dtype=float' to ensure math is fast and accurate
    I = np.array(current_infected, dtype=float)
    A = np.array(allowed_students, dtype=float)
    Risk = np.array(community_risk, dtype=float)
    alpha = np.array(alpha_m, dtype=float)
    beta_vec = np.array(beta, dtype=float)

    # 2. Vectorized Math (Replaces the nested loops)
    # Infection within own class
    in_class = alpha * I * A
    
    # Infection from community
    community = beta_vec * Risk * (A**2)
    
    # Infection from other classes (Vectorized cross-interaction)
    # Calculate infection proportion for all classes at once
    safe_A = np.maximum(A, 1e-6) # Avoid division by zero
    props = np.where(A > 0, I / safe_A, 0.0)
    
    # Calculate average prop of "others" for each agent
    total_prop_sum = np.sum(props)
    if n_classes > 1:
        avg_others = (total_prop_sum - props) / (n_classes - 1)
    else:
        avg_others = np.zeros_like(props)
        
    # Shared students calculation
    shared = (A * 0.3).astype(int) # Hardcoded 0.3 from your original code
    cross_class = delta * shared * avg_others

    # 3. Sum and Cap
    total = in_class + community + cross_class
    final = np.minimum(total, A)
    
    # 4. Return as LIST (Preserves compatibility with your other files)
    return final.astype(int).tolist()


# OLD CODE:
    
# import numpy as np

# SEED = 42
# np.random.seed(SEED)

# def simulate_infections_n_classrooms(n_classes, alpha_m, beta, delta, current_infected, allowed_students,
#                                      community_risk):
#     """
#     Updated simulation function.
#     For each classroom i:

#     I(i) = alpha_m[i] * current_infected[i] * allowed_students[i]
#            + beta[i] * community_risk[i] * allowed_students[i]^2
#            + delta * shared_students * p_i

#     where:
#       - shared_students = int(allowed_students[i] * shared_student_fraction)
#       - p_i is the average infection proportion in the other classrooms.
#     """
#     shared_student_fraction = 0.3
#     new_infected = []
#     for i in range(n_classes):
#         current_inf = current_infected[i]
#         allowed = allowed_students[i]
#         comm_risk = community_risk[i]

#         # Within-classroom infections
#         in_class_term = alpha_m[i] * current_inf * allowed

#         # Community risk infections
#         community_term = beta[i] * comm_risk * (allowed ** 2)

#         # Compute average infection proportion from the other classrooms
#         other_props = []
#         for j in range(n_classes):
#             if i != j:
#                 if allowed_students[j] > 0:
#                     other_props.append(current_infected[j] / allowed_students[j])
#                 else:
#                     other_props.append(0)
#         avg_prop = np.mean(other_props) if other_props else 0

#         # Compute number of shared students based on the fraction
#         shared_students = int(allowed * shared_student_fraction)

#         # Cross-classroom infections using the population game formulation
#         cross_class_term = delta * shared_students * avg_prop

#         total_infected = in_class_term + community_term + cross_class_term

#         # Ensure the new infections do not exceed the number of allowed students
#         total_infected = np.minimum(total_infected, allowed)
#         new_infected.append(int(total_infected))
#     return new_infected

