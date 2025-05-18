import numpy as np
import matplotlib.pyplot as plt

# Zakres wartości wejściowych
x = np.linspace(-5, 5, 1000)

# Definicja funkcji Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# Obliczanie wartości dla wykresu
y = leaky_relu(x)

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.title('Funkcja aktywacji Leaky ReLU')
plt.xlabel('Wejście (x)')
plt.ylabel('Wyjście f(x)')
plt.xlim(-5, 5)
plt.ylim(-0.5, 5)

# Dodanie opisu funkcji matematycznej - bez użycia cases
plt.text(1.5, 3.5, 'f(x) = x        dla x > 0', fontsize=14)
plt.text(1.5, 3.0, 'f(x) = αx    dla x ≤ 0', fontsize=14)
plt.text(1.5, 2.5, 'gdzie α = 0.01', fontsize=12)

# Zaznaczenie punktu przegięcia
plt.plot(0, 0, 'ro')

plt.tight_layout()
plt.show()