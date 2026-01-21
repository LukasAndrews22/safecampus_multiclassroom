import numpy as np

SEED = 42
np.random.seed(SEED)


def simulate_infections_n_classrooms(n_classes, alpha_m, beta, delta, current_infected, allowed_students,
                                     community_risk):
    """
    Updated simulation function.
    For each classroom i:

    I(i) = alpha_m[i] * current_infected[i] * allowed_students[i]
           + beta[i] * community_risk[i] * allowed_students[i]^2
           + delta * shared_students * p_i

    where:
      - shared_students = int(allowed_students[i] * shared_student_fraction)
      - p_i is the average infection proportion in the other classrooms.
    """
    shared_student_fraction = 0.3
    new_infected = []
    for i in range(n_classes):
        current_inf = current_infected[i]
        allowed = allowed_students[i]
        comm_risk = community_risk[i]

        # Within-classroom infections
        in_class_term = alpha_m[i] * current_inf * allowed

        # Community risk infections
        community_term = beta[i] * comm_risk * (allowed ** 2)

        # Compute average infection proportion from the other classrooms
        other_props = []
        for j in range(n_classes):
            if i != j:
                if allowed_students[j] > 0:
                    other_props.append(current_infected[j] / allowed_students[j])
                else:
                    other_props.append(0)
        avg_prop = np.mean(other_props) if other_props else 0

        # Compute number of shared students based on the fraction
        shared_students = int(allowed * shared_student_fraction)

        # Cross-classroom infections using the population game formulation
        cross_class_term = delta * shared_students * avg_prop

        total_infected = in_class_term + community_term + cross_class_term

        # Ensure the new infections do not exceed the number of allowed students
        total_infected = np.minimum(total_infected, allowed)
        new_infected.append(int(total_infected))
    return new_infected

