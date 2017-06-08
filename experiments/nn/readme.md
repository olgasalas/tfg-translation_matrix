Hicimos experimentos para saber qué red neuronal era la que mejores resultados obtenía con los vectores originales. Tras dichos experimentos, concluímos que la red neuronal de 2 capas era el mejor de los casos.
Por ello se han decidido obtener algunos datos más.

Se han empleado TODOS los vectores, tanto los que están en el diccionario como los que no para la obtención de los datos. Basta con pasarle dichos vectores al modelo ya entrenado con los que sí están en el diccionario.

	- Generación de 100 muestras aleatorias de TODOS los vectores, calcular su top 5, distancia de la traducción con cada traducción del top 5, y distancia con la traducción real (en caso de que tenga)
	- Generación de 1000 muestra, para el mismo experimento que en el caso anterior.
	- En función del tiempo que tarde, cálculo de la métrica anterior con TODO el corpus en inglés.

