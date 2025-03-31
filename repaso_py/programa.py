# == Zona de modulos ==


# == Zona de funciones ==

def funciones_cadenas():
    cad = "Hola"
    
    # Se itera en cada elemento de cad
    for x in cad:
        print(x+".")
    
    print("===")
    print(cad[0])
    print(cad[1])
    print(cad[2])
    print(cad[3])

    print("La cadena tiene un tamaño de:", len(cad))

    # Reemplaza la H por J
    cad.replace("H","J")
    # Pero no se muestra, porque realmente replace no modifica la cadena
    print(cad)
    # Ahora si se ve el cambio
    print(cad.replace("H","J"))

    num = 10
    # No se puede hacer -->  print(cad + num)
    # Lo que se puede hacer es lo siguiente
    print(f"El numero es {num}")
    print(f"{cad}{num}")
    # Ojo este mete un +
    print(f"{cad}+{num}")
    print(f"{cad}"+f"{num}")


def funciones_listas():
    lista = [1,2,3,"numeros"]
    # Muestra la lista al completo, incluso con corchetes
    print(lista)
    # Muestra un elemento concreto
    print(lista[0])
    # Muestra cada elemento con un bucle
    for x in lista:
        print(x)
    # Puedo mostrar el tamaño dentro de un print!
    print(f"El tamaño de la lista es {len(lista)}")

    # Se puede buscar un elemento directamente
    if "numeros" in lista:
        print("Existe numeros!")
    else:
        print("No existe numeros!")


    # Introduce en la posicion indicada SIN reemplazar. Empuja a los otros
    lista.insert(2, "Nuevo")
    print(lista)

    # Esto seria para meter algo al final, pero para eso hay otras funciones
    lista.insert(len(lista),"Auxiliar")
    print(lista)

    # La forma correcta es esta
    lista.append("ultimo")
    print(lista)

    lista2 = [10,11,12]

    lista.extend(lista2)
    print(lista)

    # Borramos un elemento en concreto
    lista.remove("Auxiliar")
    print(f"La lista borrada es:  {lista}")    

    # Borro una poscion en concreto
    nueva = [0,1,2,3]
    nueva.pop(1)
    print(nueva)
    # Si no se pone nada se borrar el ultimo
    nueva.pop()
    print(nueva)

    # Otra forma de iterar
    for i in range(len(nueva)):
        print(nueva [i])

    # Mas formas de iterar
    i = 0
    while i < len(nueva):
        print(f"Mostrando en el while: {nueva[i]}")
        i = i + 1


    # Esto hace una copia por REFERENCIA, lo que se cambie en la primera se cambia en la otra
    copia_ref = nueva
    nueva.append("Prueba")
    print(nueva)
    print(f"Soy la copia_ref: {copia_ref}")

    # Para hacerlo bien se hace copy
    copia = nueva.copy()
    nueva.remove("Prueba")
    print(nueva)
    print(f"Soy la copia: {copia}")

    list_a = [1,2]
    list_b = [3,4]
    # Esto mete en UN SOLO ELEMENTO toda la lista, incluidos los [  ]
    list_a.append(list_b) 
    print(list_a)

# == Zona de codigo ejecutable ==

# Permite establecer un "main" en el programa. Realmente se ejecuta directamente lo que haya despues de las funciones
# Esto beneficia el uso de modulos y la compatibilidad
if __name__ == "__main__":
    funciones_cadenas()
    funciones_listas()