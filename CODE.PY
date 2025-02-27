import heapq
from typing import List, Tuple, Set
import numpy as np

class RubiksRace:
    def __init__(self, initial_state: str, goal_state: str):
        # Definir el alfabeto permitido
        self.alfabeto_permitido = {'*', 'B', 'R', 'V', 'N', 'Z', 'A', ' '}
        self.initial = self.read_state(initial_state)
        self.goal = self.read_state(goal_state)
        self.moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # arriba, abajo, izquierda, derecha
        
    def read_state(self, filename: str) -> np.ndarray:
        with open(filename, 'r') as f:
            lines = f.readlines()
        state = []
        for line in lines:
            line = line.strip()
            if line:  # ignorar líneas vacías
                # Validar que no haya espacios en blanco entre letras
                if ' ' in line:
                    raise ValueError(f"El archivo {filename} no debe contener espacios en blanco entre letras.")
                row = list(line)
                state.append(row)
        state_array = np.array(state)
        
        # Validar el tamaño del estado
        if filename == "meta.txt" and state_array.shape != (3, 3):
            raise ValueError(f"El archivo {filename} debe ser de tamaño 3x3. Tamaño actual: {state_array.shape}. "
                             "Asegúrate de que todas las filas tengan la misma longitud.")
        elif filename == "inicial.txt" and state_array.shape != (5, 5):
            raise ValueError(f"El archivo {filename} debe ser de tamaño 5x5. Tamaño actual: {state_array.shape}. "
                             "Asegúrate de que todas las filas tengan la misma longitud.")

        # Validar que el archivo meta no contenga asterisco
        if filename == "meta.txt" and '*' in state_array:
            raise ValueError(f"El archivo {filename} no debe contener el asterisco ('*').")

        # Validar que el archivo inicial tenga un solo asterisco en el medio
        if filename == "inicial.txt":
            if np.count_nonzero(state_array == '*') != 1:
                raise ValueError(f"El archivo {filename} debe contener un solo asterisco ('*').")
            if state_array[2, 2] != '*':
                raise ValueError(f"El asterisco ('*') en el archivo {filename} debe estar en la posición central (2, 2).")

        # Validar el alfabeto antes de retornar
        es_valido, letra_invalida = self.validar_alfabeto(state_array)
        if not es_valido:
            raise ValueError(f"El archivo {filename} contiene la letra '{letra_invalida}' que no existe en el juego. "
                           f"Letras permitidas: {', '.join(sorted(self.alfabeto_permitido - {' '}))}")
        
        return state_array
    
    def find_empty(self, state: np.ndarray) -> Tuple[int, int]:
        pos = np.where(state == '*')
        return pos[0][0], pos[1][0]
    
    def get_manhattan_distance(self, state: np.ndarray) -> int:
        """
        Heurística: Distancia Manhattan considerando el centro 3x3 como objetivo
        """
        distance = 0
        start_row = (state.shape[0] - self.goal.shape[0]) // 2
        start_col = (state.shape[1] - self.goal.shape[1]) // 2
        
        # Calcular distancia para la región central 3x3
        for i in range(self.goal.shape[0]):
            for j in range(self.goal.shape[1]):
                if state[start_row + i, start_col + j] != '*':
                    if state[start_row + i, start_col + j] != self.goal[i,j]:
                        # Buscar la posición correcta en la meta
                        pos = np.where(self.goal == state[start_row + i, start_col + j])
                        if len(pos[0]) > 0:
                            # La distancia se calcula considerando la posición en el centro
                            dist = abs(i - pos[0][0]) + abs(j - pos[1][0])
                            distance += dist * 3
        
        # Añadir penalización si el asterisco está lejos del centro
        empty_i, empty_j = self.find_empty(state)
        center_i, center_j = state.shape[0] // 2, state.shape[1] // 2
        distance += abs(empty_i - center_i) + abs(empty_j - center_j)
        
        return distance

    def get_valid_moves(self, state: np.ndarray) -> List[int]:
        """
        Obtiene solo movimientos válidos que intercambian UNA ficha con el asterisco.
        Cada movimiento representa el intercambio del asterisco con UNA ficha adyacente:
        1: Mover la ficha de abajo hacia arriba (intercambio con *)
        2: Mover la ficha de arriba hacia abajo (intercambio con *)
        3: Mover la ficha de la derecha hacia la izquierda (intercambio con *)
        4: Mover la ficha de la izquierda hacia la derecha (intercambio con *)
        """
        valid = []
        empty_i, empty_j = self.find_empty(state)
        rows, cols = state.shape
        
        # Solo permitir movimientos que intercambien UNA ficha con el asterisco
        if empty_i < rows - 1 and state[empty_i + 1, empty_j] != ' ':  # Ficha abajo
            valid.append(1)
        if empty_i > 0 and state[empty_i - 1, empty_j] != ' ':        # Ficha arriba
            valid.append(2)
        if empty_j < cols - 1 and state[empty_i, empty_j + 1] != ' ': # Ficha derecha
            valid.append(3)
        if empty_j > 0 and state[empty_i, empty_j - 1] != ' ':       # Ficha izquierda
            valid.append(4)
            
        return valid

    def apply_move(self, state: np.ndarray, move: int) -> np.ndarray:
        """
        Aplica un solo movimiento intercambiando UNA ficha con el asterisco.
        El asterisco DEBE moverse a una posición adyacente a su posición actual.
        """
        new_state = state.copy()
        empty_i, empty_j = self.find_empty(state)
        
        # Definir la posición de la ficha a intercambiar con el asterisco
        if move == 1:  # Mover ficha de abajo hacia arriba
            ficha_i, ficha_j = empty_i + 1, empty_j
        elif move == 2:  # Mover ficha de arriba hacia abajo
            ficha_i, ficha_j = empty_i - 1, empty_j
        elif move == 3:  # Mover ficha de la derecha hacia la izquierda
            ficha_i, ficha_j = empty_i, empty_j + 1
        else:  # move == 4, Mover ficha de la izquierda hacia la derecha
            ficha_i, ficha_j = empty_i, empty_j - 1
        
        # Verificar que el movimiento sea válido
        rows, cols = state.shape
        if (0 <= ficha_i < rows and 
            0 <= ficha_j < cols and 
            abs(ficha_i - empty_i) + abs(ficha_j - empty_j) == 1):  # Asegura movimiento adyacente
            
            # Intercambiar UNA ficha con el asterisco
            new_state[empty_i, empty_j] = state[ficha_i, ficha_j]
            new_state[ficha_i, ficha_j] = '*'
            return new_state
            
        return state

    def is_valid_state_transition(self, current_state: np.ndarray, next_state: np.ndarray) -> bool:
        """
        Verifica que la transición entre estados sea válida:
        - Solo se debe mover una ficha
        - El asterisco debe moverse a una posición adyacente
        - Solo debe haber dos cambios (la ficha y el asterisco)
        """
        # Encontrar posiciones del asterisco en ambos estados
        current_empty_i, current_empty_j = self.find_empty(current_state)
        next_empty_i, next_empty_j = self.find_empty(next_state)
        
        # Verificar que el asterisco se movió a una posición adyacente
        if abs(next_empty_i - current_empty_i) + abs(next_empty_j - current_empty_j) != 1:
            return False
        
        # Verificar que solo hay dos cambios (la ficha y el asterisco)
        diff_count = np.sum(next_state != current_state)
        if diff_count != 2:
            return False
            
        return True

    def is_goal_reached(self, state: np.ndarray) -> bool:
        """Verifica si la región central 3x3 coincide con la meta"""
        # Calcular las coordenadas del centro 3x3
        start_row = (state.shape[0] - self.goal.shape[0]) // 2  # Para 5x5 sería 1
        start_col = (state.shape[1] - self.goal.shape[1]) // 2  # Para 5x5 sería 1
        end_row = start_row + self.goal.shape[0]  # Para 5x5 sería 4
        end_col = start_col + self.goal.shape[1]  # Para 5x5 sería 4
        
        # Comparar la región central 3x3 con el estado meta
        return np.array_equal(state[start_row:end_row, start_col:end_col], self.goal)

    def es_meta_valida(self, meta):
        # Crear un diccionario para contar la frecuencia de cada letra
        frecuencia = {}
        
        # Contar cada letra en el estado meta
        for fila in meta:
            for letra in fila:
                if letra != '*':  # Ignorar el espacio vacío
                    frecuencia[letra] = frecuencia.get(letra, 0) + 1
                    # Si alguna letra aparece más de 4 veces, la meta no es válida
                    if frecuencia[letra] > 4:
                        return False
        return True

    def validar_alfabeto(self, state: np.ndarray) -> Tuple[bool, str]:
        """
        Verifica que todas las letras en el estado estén en el alfabeto permitido
        Retorna: (es_valido, letra_invalida)
        """
        for fila in state:
            for letra in fila:
                if letra not in self.alfabeto_permitido:
                    return False, letra
        return True, ""

    def solve(self) -> List[Tuple[np.ndarray, int]]:
        # Verificar si la meta es válida
        if not self.es_meta_valida(self.goal):
            print("No existe solución: El estado meta contiene más de 4 repeticiones de al menos una letra")
            return None
            
        # Verificar el alfabeto en ambos estados
        es_valido_inicial, letra_invalida_inicial = self.validar_alfabeto(self.initial)
        es_valido_meta, letra_invalida_meta = self.validar_alfabeto(self.goal)
        
        if not es_valido_inicial:
            print(f"No existe solución: El estado inicial contiene la letra '{letra_invalida_inicial}' que no existe en el juego")
            return None
        if not es_valido_meta:
            print(f"No existe solución: El estado meta contiene la letra '{letra_invalida_meta}' que no existe en el juego")
            return None

        start = self.initial
        counter = 0
        frontier = [(0, 0, counter, start, [])]
        visited = set()
        output_file = open('solution.txt', 'w')
        
        while frontier:
            f, g, c, current, path = heapq.heappop(frontier)
            state_tuple = tuple(map(tuple, current))
            
            if self.is_goal_reached(current):
                output_file.write("¡Solución encontrada!\n\n")
                # Escribir estado inicial
                output_file.write("Estado Inicial:\n")
                output_file.write('\n'.join(''.join(row) for row in self.initial))
                output_file.write("\n\n")
                
                # Escribir cada movimiento con sus heurísticas
                for estado, mov in path:
                    output_file.write(f"Movimiento: {mov}\n")
                    # Calcular y mostrar las heurísticas para este estado
                    h_manhattan = self.get_manhattan_distance(estado)
                    output_file.write(f"Heurística Manhattan (distancia al centro): {h_manhattan}\n")
                    output_file.write("Estado:\n")
                    output_file.write('\n'.join(''.join(row) for row in estado))
                    output_file.write("\n\n")
                
                output_file.close()
                return path
                
            if state_tuple in visited:
                continue
                
            visited.add(state_tuple)
            
            for move in self.get_valid_moves(current):
                next_state = self.apply_move(current, move)
                next_tuple = tuple(map(tuple, next_state))
                
                if next_tuple not in visited and self.is_valid_state_transition(current, next_state):
                    g_new = g + 1
                    h = self.get_manhattan_distance(next_state)
                    counter += 1
                    
                    heapq.heappush(frontier, (g_new + h, g_new, counter, next_state, path + [(next_state, move)]))
        
        output_file.write("No se encontró solución\n")
        output_file.close()
        return None

if __name__ == "__main__":
    game = RubiksRace("inicial.txt", "meta.txt")
    solution = game.solve()
    
    if solution:
        print(f"¡Solución encontrada! ({len(solution)} movimientos)")
    else:
        print("No se encontró solución")
