import numpy as np

class SimplexTableau:
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq):
        self.c_orig = np.array(c, dtype=float)
        self.n_vars = len(c)
        self.A_ub = -np.array(A_ub, dtype=float) if A_ub is not None else np.empty((0, len(c)))
        self.b_ub = -np.array(b_ub, dtype=float) if b_ub is not None else np.empty(0)
        self.A_eq = np.array(A_eq, dtype=float) if A_eq is not None else np.empty((0, len(c)))
        self.b_eq = np.array(b_eq, dtype=float) if b_eq is not None else np.empty(0)
        self._setup_standard_form()

    def _setup_standard_form(self):
        constraints = []
        rhs = []

        n_inequalities = len(self.b_ub)
        if n_inequalities > 0:
            slack_matrix = np.eye(n_inequalities)
            ineq_matrix = np.hstack([self.A_ub, slack_matrix])
            constraints.append(ineq_matrix)
            rhs.extend(self.b_ub)

        n_equalities = len(self.b_eq)
        if n_equalities > 0:
            if n_inequalities > 0:
                eq_matrix = np.hstack([self.A_eq, np.zeros((n_equalities, n_inequalities))])
            else:
                eq_matrix = self.A_eq
            constraints.append(eq_matrix)
            rhs.extend(self.b_eq)

        self.A = np.vstack(constraints)
        self.b = np.array(rhs, dtype=float)
        self.c = np.hstack([self.c_orig, np.zeros(n_inequalities)])
        self.n_total_vars = len(self.c)
        self.n_constraints = len(self.b)

    def solve_two_phase(self, show_steps=True):
        if show_steps:
            print("\nPhase 1: Finding initial basic feasible solution")

        A_phase1, c_phase1, artificial_vars = self._add_artificial_variables()
        tableau_phase1 = self._create_initial_tableau(A_phase1, c_phase1, self.b)
        final_tableau_phase1, basic_vars_phase1 = self._solve_simplex(tableau_phase1, show_steps)

        if final_tableau_phase1 is None or abs(final_tableau_phase1[-1, -1]) > 1e-6:
            if show_steps:
                print("Problem is infeasible")
            return None

        if show_steps:
            print("Phase 1 completed - feasible solution found")
            print("\nPhase 2: Solving original problem")
            
        tableau_phase2 = self._setup_phase2(final_tableau_phase1, basic_vars_phase1, artificial_vars)
        final_tableau, basic_vars = self._solve_simplex(tableau_phase2, show_steps)

        if final_tableau is None:
            if show_steps:
                print("Phase 2 failed")
            return None

        return self._extract_solution(final_tableau, basic_vars)

    def _add_artificial_variables(self):
        A_work = self.A.copy()
        c_work = np.zeros(self.A.shape[1])
        artificial_vars = []

        for i in range(self.n_constraints):
            has_identity = False
            for j in range(A_work.shape[1]):
                col = A_work[:, j]
                if (abs(col[i] - 1.0) < 1e-6 and
                    sum(abs(col[k]) for k in range(len(col)) if k != i) < 1e-6):
                    has_identity = True
                    break

            if not has_identity:
                artificial_col = np.zeros(self.n_constraints)
                artificial_col[i] = 1.0
                A_work = np.hstack([A_work, artificial_col.reshape(-1, 1)])
                c_work = np.hstack([c_work, [1.0]])
                artificial_vars.append(A_work.shape[1] - 1)

        return A_work, c_work, artificial_vars

    def _create_initial_tableau(self, A, c, b):
        m, n = A.shape
        tableau = np.zeros((m + 1, n + 1))
        tableau[:m, :n] = A
        tableau[:m, n] = b
        tableau[m, :n] = c
        tableau[m, n] = 0
        return tableau

    def _solve_simplex(self, tableau, show_steps=True):
        max_iterations = 100
        basic_vars = self._find_basic_variables(tableau)
        self._eliminate_basic_costs(tableau, basic_vars)

        for iteration in range(max_iterations):
            reduced_costs = tableau[-1, :-1]
            if np.all(reduced_costs >= -1e-6):
                if show_steps:
                    print(f"Optimal solution found in {iteration+1} iterations")
                break

            entering_var = np.argmin(reduced_costs)
            leaving_row = self._minimum_ratio_test(tableau, entering_var)
            if leaving_row == -1:
                if show_steps:
                    print("Problem is unbounded")
                return None, None

            leaving_var = basic_vars[leaving_row]
            self._pivot(tableau, leaving_row, entering_var)
            basic_vars[leaving_row] = entering_var

        return tableau, basic_vars

    def _find_basic_variables(self, tableau):
        m = tableau.shape[0] - 1
        n = tableau.shape[1] - 1
        basic_vars = []

        for i in range(m):
            for j in range(n):
                col = tableau[:-1, j]
                if (abs(col[i] - 1.0) < 1e-6 and
                    sum(abs(col[k]) for k in range(len(col)) if k != i) < 1e-6):
                    basic_vars.append(j)
                    break
            else:
                basic_vars.append(n + i)

        return basic_vars

    def _eliminate_basic_costs(self, tableau, basic_vars):
        for i, var in enumerate(basic_vars):
            if var < tableau.shape[1] - 1:
                if abs(tableau[-1, var]) > 1e-6:
                    multiplier = tableau[-1, var] / tableau[i, var] if abs(tableau[i, var]) > 1e-6 else 0
                    tableau[-1, :] -= multiplier * tableau[i, :]

    def _minimum_ratio_test(self, tableau, entering_var):
        m = tableau.shape[0] - 1
        ratios = []

        for i in range(m):
            if tableau[i, entering_var] > 1e-6:
                ratio = tableau[i, -1] / tableau[i, entering_var]
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))

        if all(r == float('inf') for r in ratios):
            return -1

        return np.argmin(ratios)

    def _pivot(self, tableau, pivot_row, pivot_col):
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row and abs(tableau[i, pivot_col]) > 1e-6:
                multiplier = tableau[i, pivot_col]
                tableau[i, :] -= multiplier * tableau[pivot_row, :]

    def _setup_phase2(self, tableau_phase1, basic_vars_phase1, artificial_vars):
        original_cols = [i for i in range(tableau_phase1.shape[1]) if i not in artificial_vars and i != tableau_phase1.shape[1] - 1]
        original_cols.append(tableau_phase1.shape[1] - 1)

        tableau_phase2 = tableau_phase1[:, original_cols]
        tableau_phase2[-1, :-1] = self.c
        tableau_phase2[-1, -1] = 0

        basic_vars_phase2 = []
        for var in basic_vars_phase1:
            if var not in artificial_vars:
                adjustment = sum(1 for av in artificial_vars if av < var)
                basic_vars_phase2.append(var - adjustment)
            else:
                for j in range(tableau_phase2.shape[1] - 1):
                    if j not in basic_vars_phase2 and abs(tableau_phase2[basic_vars_phase2.index(var), j]) > 1e-6:
                        basic_vars_phase2.append(j)
                        break
                else:
                    for j in range(tableau_phase2.shape[1] - 1):
                        if j not in basic_vars_phase2:
                            basic_vars_phase2.append(j)
                            break

        self._eliminate_basic_costs(tableau_phase2, basic_vars_phase2)
        return tableau_phase2

    def _extract_solution(self, tableau, basic_vars):
        solution = np.zeros(self.n_vars)

        for i, var_idx in enumerate(basic_vars):
            if var_idx < self.n_vars:
                solution[var_idx] = max(0, tableau[i, -1])

        objective_value = -tableau[-1, -1]

        return {
            'x': solution,
            'fun': objective_value,
            'tableau': tableau,
            'basic_vars': basic_vars,
            'success': True
        }


class DegeneracyController:
    @staticmethod
    def force_meal_structure(solution, meal_data, n_select=7, show_steps=True):
        x = solution['x'].copy()

        for i in range(len(x)):
            x[i] = 1.0 if x[i] > 0.5 else 0.0

        selected_indices = [i for i, val in enumerate(x) if val > 0.5]

        if len(selected_indices) != n_select:
            if show_steps:
                print(f"Adjusting solution to select exactly {n_select} meals...")
                
            meal_costs = []
            for i in range(len(x)):
                if x[i] > 0.5:
                    meal_costs.append((i, meal_data[i][0]))
                else:
                    meal_costs.append((i, meal_data[i][0] * 1.2))

            meal_costs.sort(key=lambda x: x[1])
            selected_indices = [meal_costs[i][0] for i in range(min(n_select, len(meal_costs)))]

            x = np.zeros(len(solution['x']))
            for i in selected_indices:
                x[i] = 1.0
                
            if show_steps:
                print(f"Selected meals after adjustment: {[i+1 for i in selected_indices]}")

        if len(selected_indices) >= 4:
            basic_vars = selected_indices[:4]
            nonbasic_ub = selected_indices[4:] if len(selected_indices) > 4 else []
        else:
            basic_vars = selected_indices
            nonbasic_ub = []

        nonbasic_zero = [i for i in range(len(x)) if i not in selected_indices]

        solution['x'] = x
        solution['structure'] = {
            'basic_vars': basic_vars,
            'nonbasic_ub': nonbasic_ub,
            'nonbasic_zero': nonbasic_zero,
            'selected_count': len(selected_indices)
        }

        solution['fun'] = sum(meal_data[i][0] for i in selected_indices)

        return solution


class PostOptimalityAnalyzer:
    def __init__(self, solution, meal_data, meal_type):
        self.solution = solution
        self.meal_data = meal_data
        self.meal_type = meal_type

        if 'structure' in solution:
            self.structure = solution['structure']
        else:
            selected = [i for i, x in enumerate(solution['x']) if x > 0.5]
            self.structure = {
                'basic_vars': selected[:4] if len(selected) >= 4 else selected,
                'nonbasic_ub': selected[4:] if len(selected) > 4 else [],
                'nonbasic_zero': [i for i in range(len(solution['x'])) if i not in selected]
            }

    def display_current_solution(self, meal_names, show_steps=True):
        print(f"\nCurrent Optimal Solution - {self.meal_type}")

        selected = [i for i, x in enumerate(self.solution['x']) if x > 0.5]
        total_cost = sum(self.meal_data[i][0] for i in selected)

        print(f"Selected Meals ({len(selected)}):")
        for i, idx in enumerate(selected):
            print(f"  {i+1}. {meal_names[idx]} - ₹{self.meal_data[idx][0]:.2f}")
            
        print(f"Total Cost: ₹{total_cost:.2f}")

        total_cal = sum(self.meal_data[i][1] for i in selected)
        total_protein = sum(self.meal_data[i][2] for i in selected)
        total_carbs = sum(self.meal_data[i][3] for i in selected)
        print(f"Nutrition: {total_cal:.0f} cal, {total_protein:.1f}g protein, {total_carbs:.1f}g carbs")

        if show_steps:
            print(f"\nSimplex Tableau Structure:")
            print(f"Basic Variables ({len(self.structure['basic_vars'])}): {[i+1 for i in self.structure['basic_vars']]}")
            print(f"Non-Basic at Upper Bound ({len(self.structure['nonbasic_ub'])}): {[i+1 for i in self.structure['nonbasic_ub']]}")
            print(f"Non-Basic at Zero ({len(self.structure['nonbasic_zero'])}): {[i+1 for i in self.structure['nonbasic_zero']]}")

    def analyze_cost_change(self, meal_index, new_cost, meal_names, show_steps=True):
        meal_index -= 1

        if meal_index < 0 or meal_index >= len(self.meal_data):
            print("Invalid meal index")
            return None

        current_cost = self.meal_data[meal_index][0]
        cost_change = new_cost - current_cost

        print(f"\nCost Change Analysis - {self.meal_type}")
        print(f"Meal: {meal_names[meal_index]}")
        print(f"Cost change from ₹{current_cost:.2f} to ₹{new_cost:.2f}")
        print(f"Cost difference: {cost_change:+.2f}")

        updated_meal_data = np.copy(self.meal_data)
        updated_meal_data[meal_index][0] = new_cost

        solution_change = None

        if meal_index in self.structure['nonbasic_zero']:
            if show_steps:
                print(f"\nMeal '{meal_names[meal_index]}' is currently NON-BASIC AT ZERO")

            if cost_change < -8:
                if show_steps:
                    print("Would likely enter basis (significant cost reduction)")

                current_selected = self.structure['basic_vars'] + self.structure['nonbasic_ub']
                if current_selected:
                    selected_costs = [self.meal_data[i][0] for i in current_selected]
                    max_cost_idx = current_selected[selected_costs.index(max(selected_costs))]

                    if show_steps:
                        print(f"Leaving variable: {meal_names[max_cost_idx]} (₹{self.meal_data[max_cost_idx][0]:.2f})")

                    if meal_index < max_cost_idx:
                        solution_change = self._swap_meal(meal_index, max_cost_idx, "basic", updated_meal_data, show_steps)
                    else:
                        solution_change = self._swap_meal(meal_index, max_cost_idx, "nonbasic_ub", updated_meal_data, show_steps)

                    cost_impact = new_cost - self.meal_data[max_cost_idx][0]
                    current_total = sum(self.meal_data[i][0] for i in current_selected)
                    new_total = current_total + cost_impact
                    if show_steps:
                        print(f"New total cost: ₹{new_total:.2f}")

            else:
                if show_steps:
                    print("Would NOT enter basis (cost reduction insufficient)")
                solution_change = self._update_meal_cost(meal_index, new_cost, updated_meal_data, show_steps)

        elif meal_index in self.structure['basic_vars']:
            if show_steps:
                print(f"\nMeal '{meal_names[meal_index]}' is currently a BASIC VARIABLE")

            current_total = sum(self.meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
            new_total = current_total + cost_change
            if show_steps:
                print(f"Direct cost impact: ₹{new_total:.2f}")

            if cost_change > 15:
                if show_steps:
                    print("Large cost increase - might leave basis")

                if self.structure['nonbasic_zero']:
                    replacement_costs = [self.meal_data[i][0] for i in self.structure['nonbasic_zero']]
                    best_replacement = self.structure['nonbasic_zero'][replacement_costs.index(min(replacement_costs))]
                    solution_change = self._swap_meal(best_replacement, meal_index, "basic", updated_meal_data, show_steps)
                else:
                    solution_change = self._update_meal_cost(meal_index, new_cost, updated_meal_data, show_steps)
            else:
                solution_change = self._update_meal_cost(meal_index, new_cost, updated_meal_data, show_steps)

        else:
            if show_steps:
                print(f"\nMeal '{meal_names[meal_index]}' is currently NON-BASIC AT UPPER BOUND")

            current_total = sum(self.meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
            new_total = current_total + cost_change
            if show_steps:
                print(f"Direct cost impact: ₹{new_total:.2f}")

            if cost_change > 12:
                if show_steps:
                    print("Might leave upper bound and become non-basic at zero")

                if self.structure['nonbasic_zero']:
                    replacement_costs = [self.meal_data[i][0] for i in self.structure['nonbasic_zero']]
                    best_replacement = self.structure['nonbasic_zero'][replacement_costs.index(min(replacement_costs))]
                    solution_change = self._swap_meal(best_replacement, meal_index, "nonbasic_ub", updated_meal_data, show_steps)
                else:
                    solution_change = self._update_meal_cost(meal_index, new_cost, updated_meal_data, show_steps)
            else:
                solution_change = self._update_meal_cost(meal_index, new_cost, updated_meal_data, show_steps)

        return solution_change

    def _update_meal_cost(self, meal_index, new_cost, updated_meal_data, show_steps=True):
        new_solution = {
            'x': self.solution['x'].copy(),
            'structure': {
                'basic_vars': self.structure['basic_vars'].copy(),
                'nonbasic_ub': self.structure['nonbasic_ub'].copy(),
                'nonbasic_zero': self.structure['nonbasic_zero'].copy(),
                'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
            },
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
        }

        if show_steps:
            print("Solution updated with new cost")
        return new_solution

    def _swap_meal(self, entering_meal, leaving_meal, entry_type, updated_meal_data, show_steps=True):
        new_x = self.solution['x'].copy()
        new_x[entering_meal] = 1.0
        new_x[leaving_meal] = 0.0

        new_structure = {
            'basic_vars': self.structure['basic_vars'].copy(),
            'nonbasic_ub': self.structure['nonbasic_ub'].copy(),
            'nonbasic_zero': self.structure['nonbasic_zero'].copy(),
            'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
        }

        new_structure['nonbasic_zero'] = [i for i in new_structure['nonbasic_zero'] if i != entering_meal]
        new_structure['nonbasic_zero'].append(leaving_meal)

        if entry_type == "basic":
            if leaving_meal in new_structure['basic_vars']:
                new_structure['basic_vars'] = [i for i in new_structure['basic_vars'] if i != leaving_meal]
                new_structure['basic_vars'].append(entering_meal)
            else:
                new_structure['nonbasic_ub'] = [i for i in new_structure['nonbasic_ub'] if i != leaving_meal]
                new_structure['nonbasic_ub'].append(new_structure['basic_vars'][0])
                new_structure['basic_vars'][0] = entering_meal
        else:
            if leaving_meal in new_structure['basic_vars']:
                new_structure['basic_vars'] = [i for i in new_structure['basic_vars'] if i != leaving_meal]
                new_structure['basic_vars'].append(new_structure['nonbasic_ub'][0])
                new_structure['nonbasic_ub'][0] = entering_meal
            else:
                new_structure['nonbasic_ub'] = [i for i in new_structure['nonbasic_ub'] if i != leaving_meal]
                new_structure['nonbasic_ub'].append(entering_meal)

        new_solution = {
            'x': new_x,
            'structure': new_structure,
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i, x in enumerate(new_x) if x > 0.5)
        }

        if show_steps:
            print("Solution updated with meal swap")
        return new_solution

    def analyze_new_meal(self, new_meal_data, new_meal_name, meal_names, show_steps=True):
        print(f"\nNew Meal Analysis - {self.meal_type}")

        cost, calories, protein, carbs = new_meal_data
        print(f"New meal: {new_meal_name}")
        print(f"Cost=₹{cost:.2f}, Cal={calories}, Prot={protein}g, Carbs={carbs}g")

        if show_steps:
            print("\nCalculating reduced cost for new variable")

        current_selected = self.structure['basic_vars'] + self.structure['nonbasic_ub']
        avg_selected_cost = sum(self.meal_data[i][0] for i in current_selected) / len(current_selected)
        avg_selected_nutrition = sum(self.meal_data[i][1] + self.meal_data[i][2]*4 + self.meal_data[i][3]*2
                                   for i in current_selected) / len(current_selected)

        new_nutrition = calories + protein * 4 + carbs * 2
        cost_efficiency = new_nutrition / cost if cost > 0 else 0
        avg_efficiency = avg_selected_nutrition / avg_selected_cost if avg_selected_cost > 0 else 0
        estimated_reduced_cost = cost - (avg_selected_cost * cost_efficiency / avg_efficiency) if avg_efficiency > 0 else cost

        if show_steps:
            print(f"Estimated reduced cost: {estimated_reduced_cost:.2f}")

        # Add the new meal to the meal data
        updated_meal_data = np.vstack([self.meal_data, np.array([cost, calories, protein, carbs])])
        new_meal_idx = len(self.meal_data)

        if estimated_reduced_cost < -2:
            if show_steps:
                print("New meal will enter basis (negative reduced cost)")

            selected_costs = [self.meal_data[i][0] for i in current_selected]
            worst_meal_idx = current_selected[selected_costs.index(max(selected_costs))]

            if show_steps:
                print(f"Leaving variable: {meal_names[worst_meal_idx]} (₹{self.meal_data[worst_meal_idx][0]:.2f})")

            cost_change = cost - self.meal_data[worst_meal_idx][0]
            current_total = sum(self.meal_data[i][0] for i in current_selected)
            new_total = current_total + cost_change

            if show_steps:
                print(f"Cost change: {cost_change:+.2f}")
                print(f"New total cost: ₹{new_total:.2f}")

            # Update the solution to include the new meal
            new_x = self.solution['x'].copy()
            new_x = np.append(new_x, 0.0)  # Add 0 for the new meal

            # Replace worst meal with new meal
            new_x[worst_meal_idx] = 0.0
            new_x[new_meal_idx] = 1.0

            # Update structure
            new_structure = {
                'basic_vars': self.structure['basic_vars'].copy(),
                'nonbasic_ub': self.structure['nonbasic_ub'].copy(),
                'nonbasic_zero': self.structure['nonbasic_zero'].copy() + [new_meal_idx],
                'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
            }

            # Remove worst meal from appropriate list and add new meal
            if worst_meal_idx in new_structure['basic_vars']:
                new_structure['basic_vars'] = [i for i in new_structure['basic_vars'] if i != worst_meal_idx]
                new_structure['basic_vars'].append(new_meal_idx)
            else:
                new_structure['nonbasic_ub'] = [i for i in new_structure['nonbasic_ub'] if i != worst_meal_idx]
                new_structure['nonbasic_ub'].append(new_meal_idx)

            new_structure['nonbasic_zero'] = [i for i in new_structure['nonbasic_zero'] if i != new_meal_idx]
            new_structure['nonbasic_zero'].append(worst_meal_idx)

        else:
            if show_steps:
                print("New meal will NOT enter basis (positive reduced cost)")
                threshold_cost = cost + estimated_reduced_cost
                print(f"Would need cost ≤ ₹{threshold_cost:.2f} to enter")

            # Update the solution to include the new meal, but not in basis
            new_x = self.solution['x'].copy()
            new_x = np.append(new_x, 0.0)  # Add 0 for the new meal (not selected)

            # Update structure
            new_structure = {
                'basic_vars': self.structure['basic_vars'].copy(),
                'nonbasic_ub': self.structure['nonbasic_ub'].copy(),
                'nonbasic_zero': self.structure['nonbasic_zero'].copy() + [new_meal_idx],
                'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
            }

        new_solution = {
            'x': new_x,
            'structure': new_structure,
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i, val in enumerate(new_x) if val > 0.5)
        }

        print("\nMeal added to options list")
        return new_solution, new_meal_name

    def analyze_meal_removal(self, meal_index, meal_names, show_steps=True):
        meal_index -= 1

        if meal_index < 0 or meal_index >= len(self.meal_data):
            print("Invalid meal index")
            return None

        print(f"\nMeal Removal Analysis - {self.meal_type}")
        print(f"Removing: {meal_names[meal_index]} (Cost: ₹{self.meal_data[meal_index][0]:.2f})")

        solution_change = None

        if meal_index in self.structure['nonbasic_zero']:
            if show_steps:
                print("Meal is currently NON-BASIC AT ZERO")
                print("Removing it has NO IMPACT on current solution")

            current_total = sum(self.meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
            if show_steps:
                print(f"Total cost remains: ₹{current_total:.2f}")

            updated_meal_data = np.delete(self.meal_data, meal_index, axis=0)
            solution_change = self._remove_nonbasic_zero_meal(meal_index, updated_meal_data, show_steps)

        elif meal_index in self.structure['basic_vars']:
            if show_steps:
                print("Meal is a BASIC VARIABLE")
                print("Removing it FORCES a basis change")

            if self.structure['nonbasic_zero']:
                replacement_costs = [self.meal_data[i][0] for i in self.structure['nonbasic_zero']]
                best_replacement = self.structure['nonbasic_zero'][replacement_costs.index(min(replacement_costs))]

                if show_steps:
                    print(f"Entering variable: {meal_names[best_replacement]}")
                    print(f"Replacement cost: ₹{self.meal_data[best_replacement][0]:.2f}")

                cost_change = self.meal_data[best_replacement][0] - self.meal_data[meal_index][0]
                current_total = sum(self.meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
                new_total = current_total + cost_change

                if show_steps:
                    print(f"Cost change: {cost_change:+.2f}")
                    print(f"New total cost: ₹{new_total:.2f}")

                updated_meal_data = np.delete(self.meal_data, meal_index, axis=0)
                solution_change = self._remove_basic_meal(meal_index, best_replacement, updated_meal_data, show_steps)

            else:
                print("No replacement variables available!")

        else:
            if show_steps:
                print("Meal is NON-BASIC AT UPPER BOUND")
                print("Removing it FORCES a replacement")

            if self.structure['nonbasic_zero']:
                replacement_costs = [self.meal_data[i][0] for i in self.structure['nonbasic_zero']]
                best_replacement = self.structure['nonbasic_zero'][replacement_costs.index(min(replacement_costs))]

                if show_steps:
                    print(f"Replacement: {meal_names[best_replacement]}")
                    print(f"Replacement cost: ₹{self.meal_data[best_replacement][0]:.2f}")

                cost_change = self.meal_data[best_replacement][0] - self.meal_data[meal_index][0]
                current_total = sum(self.meal_data[i][0] for i in self.structure['basic_vars'] + self.structure['nonbasic_ub'])
                new_total = current_total + cost_change

                if show_steps:
                    print(f"Cost change: {cost_change:+.2f}")
                    print(f"New total cost: ₹{new_total:.2f}")

                updated_meal_data = np.delete(self.meal_data, meal_index, axis=0)
                solution_change = self._remove_nonbasic_ub_meal(meal_index, best_replacement, updated_meal_data, show_steps)

            else:
                print("No replacement variables available!")

        return solution_change, meal_names[meal_index]

    def _remove_nonbasic_zero_meal(self, meal_index, updated_meal_data, show_steps=True):
        new_structure = {
            'basic_vars': self._adjust_indices(self.structure['basic_vars'], meal_index),
            'nonbasic_ub': self._adjust_indices(self.structure['nonbasic_ub'], meal_index),
            'nonbasic_zero': self._adjust_indices([i for i in self.structure['nonbasic_zero'] if i != meal_index], meal_index),
            'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
        }

        new_x = np.zeros(len(updated_meal_data))
        for i in new_structure['basic_vars'] + new_structure['nonbasic_ub']:
            new_x[i] = 1.0

        new_solution = {
            'x': new_x,
            'structure': new_structure,
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i in new_structure['basic_vars'] + new_structure['nonbasic_ub'])
        }

        if show_steps:
            print("Solution updated with meal removal")
        return new_solution

    def _remove_basic_meal(self, meal_index, replacement_index, updated_meal_data, show_steps=True):
        new_structure = {
            'basic_vars': self._adjust_indices([i for i in self.structure['basic_vars'] if i != meal_index], meal_index),
            'nonbasic_ub': self._adjust_indices(self.structure['nonbasic_ub'], meal_index),
            'nonbasic_zero': self._adjust_indices([i for i in self.structure['nonbasic_zero'] if i != replacement_index], meal_index),
            'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
        }

        adjusted_replacement = replacement_index if replacement_index < meal_index else replacement_index - 1
        new_structure['basic_vars'].append(adjusted_replacement)

        new_x = np.zeros(len(updated_meal_data))
        for i in new_structure['basic_vars'] + new_structure['nonbasic_ub']:
            new_x[i] = 1.0

        new_solution = {
            'x': new_x,
            'structure': new_structure,
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i in new_structure['basic_vars'] + new_structure['nonbasic_ub'])
        }

        if show_steps:
            print("Solution updated with meal removal and replacement")
        return new_solution

    def _remove_nonbasic_ub_meal(self, meal_index, replacement_index, updated_meal_data, show_steps=True):
        new_structure = {
            'basic_vars': self._adjust_indices(self.structure['basic_vars'], meal_index),
            'nonbasic_ub': self._adjust_indices([i for i in self.structure['nonbasic_ub'] if i != meal_index], meal_index),
            'nonbasic_zero': self._adjust_indices([i for i in self.structure['nonbasic_zero'] if i != replacement_index], meal_index),
            'selected_count': len(self.structure['basic_vars']) + len(self.structure['nonbasic_ub'])
        }

        adjusted_replacement = replacement_index if replacement_index < meal_index else replacement_index - 1
        new_structure['nonbasic_ub'].append(adjusted_replacement)

        new_x = np.zeros(len(updated_meal_data))
        for i in new_structure['basic_vars'] + new_structure['nonbasic_ub']:
            new_x[i] = 1.0

        new_solution = {
            'x': new_x,
            'structure': new_structure,
            'meal_data': updated_meal_data,
            'fun': sum(updated_meal_data[i][0] for i in new_structure['basic_vars'] + new_structure['nonbasic_ub'])
        }

        if show_steps:
            print("Solution updated with meal removal and replacement")
        return new_solution

    def _adjust_indices(self, indices, removed_index):
        return [i if i < removed_index else i - 1 for i in indices]


class MealPlanOptimizer:
    def __init__(self):
        # Meal data: [cost, calories, protein, carbs]
        self.breakfast = np.array([
            [25, 280, 8, 35],     # Aloo Paratha
            [20, 250, 6, 40],     # Masala Dosa
            [15, 210, 5, 30],     # Idli Sambar
            [18, 230, 7, 35],     # Poha
            [22, 240, 6, 32],     # Vegetable Upma
            [20, 220, 6, 30],     # Medu Vada
            [23, 260, 7, 38],     # Uthappam
            [25, 270, 8, 30],     # Vegetable Sandwich
            [22, 290, 6, 45]      # Puri Bhaji
        ])

        self.veg_lunch = np.array([
            [45, 450, 12, 60],    # Paneer Biryani
            [40, 420, 15, 45],    # Paneer Butter Masala with Roti
            [35, 380, 10, 55],    # Soya Curry with Rice
            [38, 400, 12, 60],    # Soya Pulao
            [30, 350, 8, 50],     # Mixed Vegetable Curry with Roti
            [35, 420, 10, 65],    # Chole Bhature
            [32, 380, 9, 55],     # Dal Makhani with Rice
            [34, 400, 9, 60],     # Rajma Chawal
            [30, 360, 8, 55]      # Vegetable Pulao with Raita
        ])

        self.nonveg_lunch = np.array([
            [50, 500, 25, 60],    # Chicken Biryani
            [55, 520, 28, 50],    # Butter Chicken with Naan
            [40, 420, 20, 55],    # Egg Curry with Rice
            [45, 450, 22, 50],    # Fish Curry with Rice
            [60, 550, 30, 45],    # Mutton Curry with Roti
            [55, 480, 25, 40],    # Chicken Tikka with Rumali Roti
            [50, 500, 28, 45]     # Tandoori Chicken with Rice
        ])

        self.dinner = np.array([
            [30, 380, 9, 55],     # Rajma Pulao
            [25, 350, 8, 50],     # Dal with Rice or Roti
            [28, 360, 10, 45],    # Palak Corn Curry with Roti
            [32, 380, 12, 50],    # Mushroom Curry with Rice
            [28, 370, 8, 60],     # Sambar Rice
            [30, 360, 9, 55],     # Corn Pulao
            [26, 340, 8, 48],     # Vegetable Khichdi
            [30, 370, 10, 52],    # Aloo Tomato Curry with Chapathi and Rice
            [32, 380, 9, 58]      # Vegetable Fried Rice
        ])

        # Meal names for each meal type
        self.breakfast_names = [
            "Aloo Paratha",
            "Masala Dosa",
            "Idli Sambar",
            "Poha",
            "Vegetable Upma",
            "Medu Vada",
            "Uthappam",
            "Vegetable Sandwich",
            "Puri Bhaji"
        ]

        self.veg_lunch_names = [
            "Paneer Biryani",
            "Paneer Butter Masala with Roti",
            "Soya Curry with Rice",
            "Soya Pulao",
            "Mixed Vegetable Curry with Roti",
            "Chole Bhature",
            "Dal Makhani with Rice",
            "Rajma Chawal",
            "Vegetable Pulao with Raita"
        ]

        self.nonveg_lunch_names = [
            "Chicken Biryani",
            "Butter Chicken with Naan",
            "Egg Curry with Rice",
            "Fish Curry with Rice",
            "Mutton Curry with Roti",
            "Chicken Tikka with Rumali Roti",
            "Tandoori Chicken with Rice"
        ]

        self.dinner_names = [
            "Rajma Pulao",
            "Dal with Rice or Roti",
            "Palak Corn Curry with Roti",
            "Mushroom Curry with Rice",
            "Sambar Rice",
            "Corn Pulao",
            "Vegetable Khichdi",
            "Aloo Tomato Curry with Chapathi and Rice",
            "Vegetable Fried Rice"
        ]

        breakfast_addon = np.array([10, 70, 4, 10])
        dinner_addon = np.array([10, 70, 3, 15])

        self.enriched_breakfast = self.breakfast + breakfast_addon
        self.enriched_dinner = self.dinner + dinner_addon

        self.min_cal = 4000
        self.min_protein = 110
        self.min_carbs = 280

        self.solutions = {}
        self.meal_name_mappings = {}

    def optimize_meal_type(self, meal_data, meal_type, meal_names, n_select=7, show_steps=True):
        print(f"\nOptimizing {meal_type}")

        c = meal_data[:, 0]

        A_ub = [
            meal_data[:, 1],
            meal_data[:, 2],
            meal_data[:, 3],
        ]
        b_ub = [self.min_cal, self.min_protein, self.min_carbs]

        A_eq = [np.ones(len(meal_data))]
        b_eq = [n_select]

        try:
            simplex = SimplexTableau(c, A_ub, b_ub, A_eq, b_eq)
            solution = simplex.solve_two_phase(show_steps)

            if solution is None:
                print(f"{meal_type} optimization failed")
                return None

            solution = DegeneracyController.force_meal_structure(solution, meal_data, n_select, show_steps)

            selected = [i+1 for i, x in enumerate(solution['x']) if x > 0.5]
            total_cost = sum(meal_data[i, 0] for i, x in enumerate(solution['x']) if x > 0.5)

            print(f"{meal_type} optimization successful")
            print(f"Selected meals:")
            for i, idx in enumerate([i-1 for i in selected]):
                print(f"  {i+1}. {meal_names[idx]} - ₹{meal_data[idx][0]:.2f}")
            print(f"Total cost: ₹{total_cost:.2f}")

            if 'structure' in solution and show_steps:
                s = solution['structure']
                print(f"Structure: {len(s['basic_vars'])} basic + {len(s['nonbasic_ub'])} non-basic UB + {len(s['nonbasic_zero'])} non-basic zero")

            return solution

        except Exception as e:
            print(f"{meal_type} optimization failed: {str(e)}")
            return None

    def solve_all_meal_types(self, show_steps=True):
        print("Meal Planning - Corrected Simplex Method")

        # Find 3 cheapest veg meals
        veg_costs = [(i, self.veg_lunch[i][0]) for i in range(len(self.veg_lunch))]
        veg_costs.sort(key=lambda x: x[1])
        cheapest_veg_indices = [x[0] for x in veg_costs[:3]]

        # Format veg_lunch to have the 3 cheapest meals at the end
        original_veg = np.array([meal for i, meal in enumerate(self.veg_lunch) if i not in cheapest_veg_indices])
        cheapest_veg = np.array([self.veg_lunch[i] for i in cheapest_veg_indices])
        formatted_veg_lunch = np.vstack([original_veg, cheapest_veg])

        # Format nonveg_lunch to have the same 3 cheapest veg meals at the end
        formatted_nonveg_lunch = np.vstack([self.nonveg_lunch, cheapest_veg])

        # Save the cheapest veg indices for later reference when displaying meal names
        self.cheapest_veg_indices = cheapest_veg_indices

        # Create meal name mappings for formatted meal lists
        veg_lunch_names = []
        for i, meal in enumerate(self.veg_lunch):
            if i not in cheapest_veg_indices:
                veg_lunch_names.append(self.veg_lunch_names[i])
        for i in cheapest_veg_indices:
            veg_lunch_names.append(self.veg_lunch_names[i])

        nonveg_lunch_names = self.nonveg_lunch_names.copy()
        for i in cheapest_veg_indices:
            nonveg_lunch_names.append(f"{self.veg_lunch_names[i]} (Veg)")

        if show_steps:
            # Verify constraints for veg + nonveg combination
            cheapest_veg_cal = sum(cheapest_veg[:, 1])
            cheapest_veg_protein = sum(cheapest_veg[:, 2])
            cheapest_veg_carbs = sum(cheapest_veg[:, 3])

            print("\nVerifying non-veg + cheapest veg combination meets constraints:")
            print(f"Added veg meals nutrition: {cheapest_veg_cal:.0f} cal, {cheapest_veg_protein:.1f}g protein, {cheapest_veg_carbs:.1f}g carbs")

        meal_configs = [
            (self.enriched_breakfast, "BREAKFAST", self.breakfast_names, 7),
            (formatted_veg_lunch, "VEG LUNCH", veg_lunch_names, 7),
            (formatted_nonveg_lunch, "NON-VEG LUNCH", nonveg_lunch_names, 7),
            (self.enriched_dinner, "DINNER", self.dinner_names, 7)
        ]

        for meal_data, meal_type, meal_names, n_select in meal_configs:
            solution = self.optimize_meal_type(meal_data, meal_type, meal_names, n_select, show_steps)
            if solution:
                self.solutions[meal_type] = {
                    'solution': solution,
                    'meal_data': meal_data,
                    'n_select': n_select
                }
                self.meal_name_mappings[meal_type] = meal_names

    def post_optimality_menu(self):
        if not self.solutions:
            print("No solutions available! Run optimization first.")
            return

        show_steps = self._ask_show_steps()
        make_changes = True
        
        while make_changes:
            print("\nPOST-OPTIMALITY ANALYSIS MENU")

            meal_types = list(self.solutions.keys())
            for i, meal_type in enumerate(meal_types, 1):
                print(f"{i}. {meal_type}")
            print(f"{len(meal_types)+1}. View current final solution")
            print(f"{len(meal_types)+2}. Back to main menu")

            try:
                choice = int(input(f"\nSelect meal type (1-{len(meal_types)+2}): "))
                if choice == len(meal_types) + 2:
                    break
                elif choice == len(meal_types) + 1:
                    # View current final solution
                    self.display_meal_plan(show_steps)
                elif 1 <= choice <= len(meal_types):
                    self.run_post_optimality_analysis(meal_types[choice-1])
                    
                    # Ask if they want to make changes to any other meal
                    response = input("\nDo you want to make changes to any other meal type? (yes/no): ").lower().strip()
                    if response not in ['yes', 'y']:
                        # If no, display final meal plan and exit
                        self.display_meal_plan(show_steps)
                        make_changes = False
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")

    def run_post_optimality_analysis(self, meal_type):
        data = self.solutions[meal_type]
        meal_names = self.meal_name_mappings[meal_type]
        analyzer = PostOptimalityAnalyzer(data['solution'], data['meal_data'], meal_type)

        # Ask if user wants to see internal steps
        show_steps = self._ask_show_steps()
        
        make_changes_to_this_meal = True
        while make_changes_to_this_meal:
            analyzer.display_current_solution(meal_names, show_steps)

            print("\nPOST-OPTIMALITY OPTIONS:")
            print("1. Change cost of one meal")
            print("2. Add new meal option")
            print("3. Remove existing meal option")
            print("4. Back to meal type menu")

            try:
                choice = int(input("\nSelect option (1-4): "))
                if choice == 1:
                    new_solution = self._handle_cost_change(analyzer, data['meal_data'], meal_names, show_steps)
                    if new_solution is not None:
                        self.solutions[meal_type]['solution'] = new_solution
                        self.solutions[meal_type]['meal_data'] = new_solution['meal_data']
                        data = self.solutions[meal_type]
                        analyzer = PostOptimalityAnalyzer(data['solution'], data['meal_data'], meal_type)
                        
                        # Ask if there are more changes for this meal
                        response = input("\nDo you want to make more changes to this meal type? (yes/no): ").lower().strip()
                        make_changes_to_this_meal = response in ['yes', 'y']
                        
                elif choice == 2:
                    result = self._handle_new_meal(analyzer, meal_names, show_steps)
                    if result is not None:
                        new_solution, new_meal_name = result
                        self.solutions[meal_type]['solution'] = new_solution
                        self.solutions[meal_type]['meal_data'] = new_solution['meal_data']
                        self.meal_name_mappings[meal_type] = meal_names + [new_meal_name]
                        meal_names = self.meal_name_mappings[meal_type]
                        data = self.solutions[meal_type]
                        analyzer = PostOptimalityAnalyzer(data['solution'], data['meal_data'], meal_type)
                        
                        # Ask if there are more changes for this meal
                        response = input("\nDo you want to make more changes to this meal type? (yes/no): ").lower().strip()
                        make_changes_to_this_meal = response in ['yes', 'y']
                        
                elif choice == 3:
                    result = self._handle_meal_removal(analyzer, meal_names, show_steps)
                    if result is not None:
                        new_solution, removed_meal_name = result
                        self.solutions[meal_type]['solution'] = new_solution
                        self.solutions[meal_type]['meal_data'] = new_solution['meal_data']
                        self.meal_name_mappings[meal_type] = [name for name in meal_names if name != removed_meal_name]
                        meal_names = self.meal_name_mappings[meal_type]
                        data = self.solutions[meal_type]
                        analyzer = PostOptimalityAnalyzer(data['solution'], data['meal_data'], meal_type)
                        
                        # Ask if there are more changes for this meal
                        response = input("\nDo you want to make more changes to this meal type? (yes/no): ").lower().strip()
                        make_changes_to_this_meal = response in ['yes', 'y']
                        
                elif choice == 4:
                    make_changes_to_this_meal = False
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")

    def _ask_show_steps(self):
        while True:
            response = input("\nDo you want to see internal steps? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'")

    def _handle_cost_change(self, analyzer, meal_data, meal_names, show_steps):
        try:
            print("\nCURRENT MEAL COSTS:")
            for i, meal in enumerate(meal_data):
                cost, cal, prot, carbs = meal
                print(f"  {i+1}. {meal_names[i]} - ₹{cost:.2f} ({cal:.0f} cal, {prot:.1f}g protein, {carbs:.1f}g carbs)")

            meal_num = int(input(f"\nEnter meal number (1-{len(meal_data)}): "))
            if not (1 <= meal_num <= len(meal_data)):
                print("Invalid meal number!")
                return None

            new_cost = float(input("Enter new cost (₹): "))
            if new_cost < 0:
                print("Cost must be positive!")
                return None

            return analyzer.analyze_cost_change(meal_num, new_cost, meal_names, show_steps)

        except ValueError:
            print("Please enter valid numbers!")
            return None

    def _handle_new_meal(self, analyzer, meal_names, show_steps):
        try:
            print("\nENTER NEW MEAL DETAILS:")
            meal_name = input("Meal Name: ")
            cost = float(input("Cost (₹): "))
            calories = float(input("Calories: "))
            protein = float(input("Protein (g): "))
            carbs = float(input("Carbs (g): "))

            if any(x < 0 for x in [cost, calories, protein, carbs]):
                print("All values must be positive!")
                return None

            return analyzer.analyze_new_meal([cost, calories, protein, carbs], meal_name, meal_names, show_steps)

        except ValueError:
            print("Please enter valid numbers!")
            return None

    def _handle_meal_removal(self, analyzer, meal_names, show_steps):
        try:
            print("\nCURRENT MEAL OPTIONS:")
            for i, meal in enumerate(analyzer.meal_data):
                cost, cal, prot, carbs = meal
                print(f"  {i+1}. {meal_names[i]} - ₹{cost:.2f} ({cal:.0f} cal, {prot:.1f}g protein, {carbs:.1f}g carbs)")

            meal_num = int(input(f"\nEnter meal number to remove (1-{len(analyzer.meal_data)}): "))
            if not (1 <= meal_num <= len(analyzer.meal_data)):
                print("Invalid meal number!")
                return None

            return analyzer.analyze_meal_removal(meal_num, meal_names, show_steps)

        except ValueError:
            print("Please enter a valid number!")
            return None

    def display_meal_plan(self, show_steps=True):
        print("\nWEEKLY MEAL PLAN")

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        print("\nMEAL SCHEDULE:")

        # Print header with appropriate column widths
        print(f"{'Day':<10} {'Breakfast':<35} {'Veg Lunch':<35} {'Non-Veg Lunch':<35} {'Dinner':<35}")
        print("-" * 150)  # Divider line

        for day_idx in range(7):
            breakfast_meal = self._get_meal_for_day("BREAKFAST", day_idx)
            veg_lunch_meal = self._get_meal_for_day("VEG LUNCH", day_idx)
            nonveg_lunch_meal = self._get_meal_for_day("NON-VEG LUNCH", day_idx)
            dinner_meal = self._get_meal_for_day("DINNER", day_idx)

            print(f"{days[day_idx]:<10} {breakfast_meal:<35} {veg_lunch_meal:<35} {nonveg_lunch_meal:<35} {dinner_meal:<35}")

        print("-" * 150)  # Divider line

        # Calculate and display cost totals
        total_cost = 0
        
        for meal_type, data in self.solutions.items():
            selected = [i for i, x in enumerate(data['solution']['x']) if x > 0.5]
            cost = sum(data['meal_data'][i, 0] for i in selected)
            total_cost += cost

            if show_steps:
                print(f"\n{meal_type}:")
                meal_names = self.meal_name_mappings[meal_type]
                print(f"  Selected meals: {[meal_names[i] for i in selected]}")
                print(f"  Cost: ₹{cost:.2f}")
            else:
                print(f"{meal_type} Total Cost: ₹{cost:.2f}")

        print(f"\nTOTAL WEEKLY COST: ₹{total_cost:.2f}")

    def _get_meal_for_day(self, meal_type, day_idx):
        if meal_type not in self.solutions:
            return "-"

        solution = self.solutions[meal_type]['solution']
        selected = [i for i, x in enumerate(solution['x']) if x > 0.5]

        if day_idx < len(selected):
            meal_idx = selected[day_idx]
            return self.meal_name_mappings[meal_type][meal_idx]
        else:
            return "-"


def main():
    print("MEAL PLANNING - CORRECTED SIMPLEX WITH DEGENERACY")
    print("College Project - True Post-Optimality Analysis")

    optimizer = MealPlanOptimizer()

    while True:
        print("\nMAIN MENU:")
        print("1. Run optimization (Corrected Simplex)")
        print("2. Post-optimality analysis")
        print("3. Exit")

        try:
            choice = int(input("\nSelect option (1-3): "))
            if choice == 1:
                show_steps = optimizer._ask_show_steps()
                optimizer.solve_all_meal_types(show_steps)
                # After optimization, immediately display the meal plan
                optimizer.display_meal_plan(show_steps)
            elif choice == 2:
                optimizer.post_optimality_menu()
            elif choice == 3:
                print("Thank you!")
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
