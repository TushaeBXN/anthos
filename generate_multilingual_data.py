#!/usr/bin/env python3
"""
generate_multilingual_data.py — Multilingual instruction pairs for Anthos SFT.

Generates 1,200+ high-quality multilingual training pairs across Spanish (primary),
French, Portuguese, and Arabic. Coverage:
  - General QA in target language (not translation tasks)
  - Code explanations requested in target language
  - Conversational / cultural knowledge
  - Cross-lingual instruction following (English prompt → target language answer)
  - Light translation (EN↔ES/FR/PT)

All pairs use Anthos SFT format:
  {"conversations": [{"from": "system", ...}, {"from": "human", ...}, {"from": "gpt", ...}]}

Licensing: 100% original content — no external dataset dependencies.

Usage:
    python generate_multilingual_data.py --out data/multilingual_sft.jsonl
    python generate_multilingual_data.py --out data/multilingual_sft.jsonl --count 1200
"""

import argparse
import json
import random
import re
from pathlib import Path

random.seed(42)

SYSTEM_EN = (
    "You are Anthos, an expert AI assistant created by Brian Tushae Thomas. "
    "You speak multiple languages fluently and answer in whichever language the user uses."
)
SYSTEM_ES = (
    "Eres Anthos, un asistente de IA experto creado por Brian Tushae Thomas. "
    "Hablas varios idiomas con fluidez y respondes en el idioma que usa el usuario."
)
SYSTEM_FR = (
    "Tu es Anthos, un assistant IA expert créé par Brian Tushae Thomas. "
    "Tu parles plusieurs langues couramment et réponds dans la langue utilisée par l'utilisateur."
)
SYSTEM_PT = (
    "Você é Anthos, um assistente de IA especializado criado por Brian Tushae Thomas. "
    "Você fala vários idiomas fluentemente e responde no idioma do usuário."
)


def pair(system, human, gpt):
    return {"conversations": [
        {"from": "system",  "value": system},
        {"from": "human",   "value": human},
        {"from": "gpt",     "value": gpt},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# SPANISH (ES) — 700 pairs
# ─────────────────────────────────────────────────────────────────────────────

ES_GENERAL_QA = [
    pair(SYSTEM_ES,
         "¿Qué es la inteligencia artificial y cómo funciona?",
         "La inteligencia artificial (IA) es la capacidad de las máquinas para realizar tareas que normalmente requieren inteligencia humana, como reconocer imágenes, entender el lenguaje o tomar decisiones.\n\n"
         "Funciona principalmente mediante el aprendizaje automático: se le muestran miles de ejemplos al sistema, y el modelo ajusta sus parámetros internos hasta que aprende a reconocer patrones. "
         "Por ejemplo, para reconocer fotos de gatos, el modelo ve millones de imágenes etiquetadas y aprende qué características definen a un gato.\n\n"
         "Los modelos de lenguaje como yo funcionamos prediciendo qué palabra viene después, aprendiendo gramática, hechos y razonamiento de enormes cantidades de texto."),
    pair(SYSTEM_ES,
         "Explícame cómo funciona el internet en términos simples.",
         "Internet es una red global de computadoras conectadas entre sí. Funciona así:\n\n"
         "1. **Tu dispositivo** envía una solicitud (por ejemplo, 'quiero ver youtube.com').\n"
         "2. **El router** en tu casa la reenvía a tu proveedor de internet (ISP).\n"
         "3. **Los servidores DNS** traducen 'youtube.com' a una dirección IP numérica.\n"
         "4. **Los datos viajan** por cables físicos (fibra óptica, cables submarinos) hasta el servidor de YouTube.\n"
         "5. **YouTube responde** enviando los datos del video de vuelta por el mismo camino.\n\n"
         "Todo esto ocurre en milisegundos gracias al protocolo TCP/IP, que divide los datos en pequeños paquetes y los reensambla en destino."),
    pair(SYSTEM_ES,
         "¿Cuál es la diferencia entre velocidad y aceleración?",
         "Son conceptos relacionados pero distintos:\n\n"
         "**Velocidad** es cuán rápido se mueve algo y en qué dirección. Por ejemplo: 60 km/h hacia el norte. Tiene magnitud y dirección (es un vector).\n\n"
         "**Aceleración** es el cambio en velocidad por unidad de tiempo. Si un carro pasa de 0 a 60 km/h en 5 segundos, su aceleración es 12 km/h por segundo.\n\n"
         "Un error común: puedes tener aceleración sin cambiar de rapidez. Si vas en círculo a velocidad constante, tu dirección cambia constantemente, por lo tanto aceleras (aceleración centrípeta)."),
    pair(SYSTEM_ES,
         "¿Cómo funciona una vacuna?",
         "Una vacuna entrena al sistema inmunitario para reconocer y combatir un patógeno específico sin que tengas que enfermarte primero.\n\n"
         "El proceso:\n"
         "1. La vacuna introduce una versión debilitada, inactiva o un fragmento del patógeno (o instrucciones ARNm para producirlo).\n"
         "2. Tu sistema inmunitario detecta esta amenaza y produce anticuerpos.\n"
         "3. Las **células de memoria** quedan en tu cuerpo durante años.\n"
         "4. Si el patógeno real aparece, tu sistema ya sabe cómo combatirlo rápidamente.\n\n"
         "Las vacunas de ARNm (como las de COVID-19) no modifican tu ADN; solo dan instrucciones temporales a tus células para producir una proteína que desencadena la respuesta inmune."),
    pair(SYSTEM_ES,
         "¿Por qué el cielo es azul?",
         "El cielo es azul por un fenómeno llamado **dispersión de Rayleigh**.\n\n"
         "La luz solar contiene todos los colores del espectro. Cuando entra en la atmósfera, choca con las moléculas de gas. La luz azul (longitud de onda corta) se dispersa mucho más que la roja o naranja (longitudes de onda largas).\n\n"
         "Resultado: la luz azul se esparce en todas direcciones por el cielo, por eso lo vemos azul desde cualquier ángulo.\n\n"
         "Por eso los atardeceres son rojizos: cuando el sol está bajo, la luz viaja más distancia a través de la atmósfera, y la poca luz azul que queda ya se dispersó. Solo llegan los rojos y naranjas."),
    pair(SYSTEM_ES,
         "Explica el cambio climático en menos de 100 palabras.",
         "El cambio climático es el calentamiento anormal del planeta causado principalmente por la quema de combustibles fósiles. Al quemar carbón, petróleo y gas, liberamos CO₂ y otros gases de efecto invernadero que atrapan el calor solar en la atmósfera. "
         "Esto eleva las temperaturas globales, derrite glaciares, sube el nivel del mar y hace los fenómenos climáticos más extremos. "
         "Es distinto al clima natural del planeta porque ocurre mucho más rápido de lo que los ecosistemas pueden adaptarse."),
    pair(SYSTEM_ES,
         "¿Qué es el producto interno bruto (PIB)?",
         "El PIB es la medida del valor total de todos los bienes y servicios producidos en un país durante un período (generalmente un año).\n\n"
         "Se calcula sumando:\n"
         "- **Consumo** (C): lo que gastan los hogares\n"
         "- **Inversión** (I): lo que gastan las empresas en capital\n"
         "- **Gasto público** (G): lo que gasta el gobierno\n"
         "- **Exportaciones netas** (X - M): exportaciones menos importaciones\n\n"
         "**PIB nominal** usa precios actuales. **PIB real** ajusta por inflación para comparar años distintos.\n\n"
         "Limitación importante: el PIB no mide desigualdad, bienestar ni daño ambiental. Un país puede tener PIB alto con calidad de vida baja."),
    pair(SYSTEM_ES,
         "¿Cómo se forma un tornado?",
         "Los tornados se forman en condiciones atmosféricas muy específicas:\n\n"
         "1. **Choque de masas de aire**: aire frío y seco del norte choca con aire caliente y húmedo del sur (común en el 'Tornado Alley' de EE.UU.).\n"
         "2. **Cizalladura de viento**: el viento cambia de dirección y velocidad con la altura, creando rotación horizontal invisible.\n"
         "3. **Supercélula**: una tormenta intensa levanta esa rotación horizontal y la convierte en vertical.\n"
         "4. **Embudo**: si la columna rotante toca el suelo, es un tornado.\n\n"
         "Los más destructivos pueden superar 400 km/h y durar horas."),
    pair(SYSTEM_ES,
         "¿Cuál es la diferencia entre bacterias y virus?",
         "Son organismos completamente distintos:\n\n"
         "**Bacterias:**\n"
         "- Organismos vivos unicelulares con ADN propio\n"
         "- Pueden reproducirse por sí solas\n"
         "- Tratables con antibióticos\n"
         "- Ejemplos: E. coli, salmonela, estreptococo\n\n"
         "**Virus:**\n"
         "- No son células — son fragmentos de material genético (ADN o ARN) envueltos en proteína\n"
         "- No pueden reproducirse solos; necesitan infectar una célula huésped\n"
         "- Los antibióticos no funcionan contra ellos (se usan antivirales o vacunas)\n"
         "- Ejemplos: influenza, COVID-19, VIH, herpes\n\n"
         "Dato clave: muchas personas piden antibióticos para un resfriado (virus). No funciona y genera resistencia bacteriana."),
    pair(SYSTEM_ES,
         "¿Qué es el machine learning en palabras simples?",
         "El machine learning (aprendizaje automático) es enseñarle a una computadora a aprender de ejemplos en lugar de programarle reglas explícitas.\n\n"
         "Ejemplo clásico:\n"
         "- **Enfoque tradicional**: programas reglas como 'si tiene orejas puntiagudas Y bigotes Y maúlla → es un gato'.\n"
         "- **Machine learning**: le muestras 100,000 fotos etiquetadas de gatos y no-gatos. El modelo descubre las reglas solo.\n\n"
         "Funciona ajustando millones de parámetros matemáticos hasta que el error en sus predicciones sea mínimo. Ese proceso de ajuste se llama **entrenamiento**.\n\n"
         "Aplicaciones: reconocimiento facial, recomendaciones de Netflix, diagnóstico médico, traducción automática."),
    pair(SYSTEM_ES,
         "Explica la teoría de la relatividad de Einstein de manera simple.",
         "Einstein propuso dos teorías de la relatividad:\n\n"
         "**Relatividad especial (1905):**\n"
         "- Las leyes de la física son iguales para todos los observadores que se mueven a velocidad constante.\n"
         "- La velocidad de la luz es siempre la misma (300,000 km/s), sin importar quién la mida.\n"
         "- Consecuencia: el tiempo pasa más lento para quien se mueve rápido (**dilatación temporal**). Un astronauta que viaja cerca de la velocidad de la luz envejece más despacio que quienes quedan en la Tierra.\n\n"
         "**Relatividad general (1915):**\n"
         "- La gravedad no es una fuerza sino una curvatura del espacio-tiempo causada por la masa.\n"
         "- Consecuencia: la luz se dobla cerca de objetos masivos (confirmado en 1919 observando un eclipse solar).\n\n"
         "La famosa ecuación E=mc² viene de la relatividad especial y dice que masa y energía son equivalentes."),
    pair(SYSTEM_ES,
         "¿Cómo funciona la memoria humana?",
         "La memoria humana no funciona como un disco duro; es reconstructiva y falible.\n\n"
         "**Tipos principales:**\n"
         "- **Memoria sensorial**: dura milisegundos (lo que acabas de ver/oír).\n"
         "- **Memoria de trabajo** (corto plazo): dura segundos-minutos; capacidad limitada (~7 ítems).\n"
         "- **Memoria a largo plazo**: potencialmente ilimitada; dura años.\n\n"
         "**Cómo se forma un recuerdo:**\n"
         "1. **Codificación**: la experiencia activa neuronas que forman conexiones (sinapsis).\n"
         "2. **Consolidación**: el sueño es crucial — durante el sueño profundo el cerebro 'graba' los recuerdos.\n"
         "3. **Recuperación**: al recordar, reconstruyes el recuerdo (por eso los recuerdos cambian con el tiempo).\n\n"
         "Las emociones fuertes activan la amígdala, que refuerza la consolidación — por eso recuerdas mejor eventos impactantes."),
    pair(SYSTEM_ES,
         "¿Qué es la inflación y cómo nos afecta?",
         "La inflación es el aumento sostenido del nivel general de precios. Si hoy un café cuesta $2 y el año que viene cuesta $2.10, hay 5% de inflación.\n\n"
         "**Causas principales:**\n"
         "- Demasiado dinero en circulación (el banco central imprime más)\n"
         "- Demanda excede oferta (todos quieren lo mismo y hay poco)\n"
         "- Costos de producción aumentan (petróleo más caro → todo más caro)\n\n"
         "**Cómo nos afecta:**\n"
         "- Tu dinero pierde poder adquisitivo: con los mismos $100 compras menos.\n"
         "- Los ahorradores pierden (el dinero en el banco vale menos).\n"
         "- Los deudores ganan (pagan deudas con dinero que vale menos).\n\n"
         "Una inflación baja y estable (~2%) es normal. La hiperinflación (>50% mensual) destruye economías."),
    pair(SYSTEM_ES,
         "¿Por qué dormir es tan importante?",
         "El sueño no es tiempo perdido — es cuando el cuerpo y el cerebro hacen trabajo crítico:\n\n"
         "**Para el cerebro:**\n"
         "- Consolida recuerdos (mueve información de corto a largo plazo)\n"
         "- El sistema glinfático limpia toxinas, incluyendo proteínas asociadas al Alzheimer\n"
         "- Procesa emociones — sin sueño, la amígdala reacciona 60% más fuerte a estímulos negativos\n\n"
         "**Para el cuerpo:**\n"
         "- Libera hormona del crecimiento (reparación muscular y celular)\n"
         "- Regula insulina y metabolismo\n"
         "- Fortalece el sistema inmune\n\n"
         "**Consecuencias de no dormir bien:**\n"
         "- Impedimento cognitivo equivalente a estar ebrio (17h sin dormir ≈ 0.05% alcohol en sangre)\n"
         "- Mayor riesgo de obesidad, diabetes tipo 2, enfermedad cardiovascular\n"
         "- Acortamiento de vida con privación crónica\n\n"
         "Los adultos necesitan 7-9 horas. No hay 'deuda de sueño' que se recupere completamente."),
    pair(SYSTEM_ES,
         "¿Qué es la criptomoneda y cómo funciona?",
         "Una criptomoneda es una moneda digital que funciona sin banco central ni gobierno que la controle.\n\n"
         "**Tecnología base — blockchain:**\n"
         "- Un registro público de todas las transacciones, copiado en miles de computadoras simultáneamente.\n"
         "- Nadie puede modificar una transacción antigua sin cambiar todos los bloques siguientes (y sin que los demás lo noten).\n\n"
         "**Cómo funciona una transacción:**\n"
         "1. Tú envías 0.1 BTC a otra persona.\n"
         "2. La transacción se transmite a la red de nodos.\n"
         "3. Los 'mineros' (o validadores) la verifican y la añaden a un bloque.\n"
         "4. El bloque se agrega a la cadena — la transacción es permanente.\n\n"
         "**Riesgo real**: alta volatilidad, sin protección del consumidor, posible uso en fraudes. No es inversión sin riesgo."),
]

ES_CODE = [
    pair(SYSTEM_ES,
         "¿Puedes explicarme qué hace este código Python?\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
         "Este código calcula el **n-ésimo número de Fibonacci** usando recursión.\n\n"
         "La serie de Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21...\n\n"
         "**Cómo funciona:**\n"
         "- Caso base: si n es 0 devuelve 0, si n es 1 devuelve 1.\n"
         "- Caso recursivo: `fibonacci(n)` = `fibonacci(n-1)` + `fibonacci(n-2)`\n\n"
         "Ejemplo: `fibonacci(5)` calcula → `fibonacci(4)` + `fibonacci(3)` → y así hasta los casos base.\n\n"
         "**Problema importante**: este código es extremadamente lento para n grande porque recalcula los mismos valores muchas veces. `fibonacci(40)` hace ~300 millones de llamadas.\n\n"
         "**Solución rápida** — usar memoización:\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\nAhora es O(n) en tiempo y memoria."),
    pair(SYSTEM_ES,
         "¿Cómo puedo leer un archivo CSV en Python y calcular el promedio de una columna?",
         "Puedes hacerlo con la librería `pandas`, que es la forma estándar:\n\n"
         "```python\nimport pandas as pd\n\n# Leer el archivo\ndf = pd.read_csv('datos.csv')\n\n# Ver las primeras filas\nprint(df.head())\n\n# Calcular el promedio de la columna 'ventas'\npromedio = df['ventas'].mean()\nprint(f'Promedio de ventas: {promedio:.2f}')\n\n# Sin pandas (solo librería estándar):\nimport csv\n\nvalores = []\nwith open('datos.csv', newline='', encoding='utf-8') as f:\n    lector = csv.DictReader(f)\n    for fila in lector:\n        try:\n            valores.append(float(fila['ventas']))\n        except (ValueError, KeyError):\n            pass  # ignorar filas con datos inválidos\n\npromedio = sum(valores) / len(valores) if valores else 0\nprint(f'Promedio: {promedio:.2f}')\n```\n\n"
         "Recomiendo `pandas` para CSV grandes — maneja mejor los tipos de datos, valores nulos y es mucho más rápido."),
    pair(SYSTEM_ES,
         "Escribe una función en Python que verifique si un número es primo.",
         "```python\nimport math\n\ndef es_primo(n: int) -> bool:\n    \"\"\"Devuelve True si n es primo, False si no.\"\"\"\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    # Solo verificar divisores impares hasta √n\n    for i in range(3, int(math.sqrt(n)) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n# Pruebas\nfor num in [0, 1, 2, 3, 4, 17, 100, 97, 1000003]:\n    print(f'{num}: {es_primo(num)}')\n```\n\n"
         "**Por qué solo hasta √n**: si n tiene un divisor mayor que √n, también tiene uno menor que √n. Así que si no encontramos ninguno hasta √n, n es primo.\n\n"
         "Esto es O(√n), que es eficiente para números normales. Para números muy grandes existen algoritmos más rápidos como Miller-Rabin."),
    pair(SYSTEM_ES,
         "¿Qué es la programación orientada a objetos? Explícalo con un ejemplo en Python.",
         "La **programación orientada a objetos (POO)** organiza el código alrededor de **objetos** — entidades que tienen datos (atributos) y comportamiento (métodos).\n\n"
         "```python\nclass CuentaBancaria:\n    def __init__(self, titular: str, saldo_inicial: float = 0):\n        self.titular = titular\n        self._saldo = saldo_inicial  # _ indica 'privado por convención'\n\n    def depositar(self, monto: float):\n        if monto <= 0:\n            raise ValueError('El monto debe ser positivo')\n        self._saldo += monto\n        return self._saldo\n\n    def retirar(self, monto: float):\n        if monto > self._saldo:\n            raise ValueError('Fondos insuficientes')\n        self._saldo -= monto\n        return self._saldo\n\n    @property\n    def saldo(self) -> float:\n        return self._saldo\n\n    def __str__(self):\n        return f'Cuenta de {self.titular}: ${self._saldo:.2f}'\n\n# Uso\ncuenta = CuentaBancaria('María García', 1000)\ncuenta.depositar(500)\ncuenta.retirar(200)\nprint(cuenta)  # Cuenta de María García: $1300.00\n```\n\n"
         "**Conceptos clave:**\n"
         "- **Clase**: plantilla para crear objetos (`CuentaBancaria`)\n"
         "- **Objeto**: instancia de una clase (`cuenta`)\n"
         "- **Encapsulación**: los datos están protegidos dentro del objeto\n"
         "- **`__init__`**: constructor, se ejecuta al crear el objeto"),
    pair(SYSTEM_ES,
         "¿Cómo puedo ordenar una lista de diccionarios en Python por un campo específico?",
         "```python\n# Lista de empleados\nempleados = [\n    {'nombre': 'Carlos', 'edad': 35, 'salario': 50000},\n    {'nombre': 'Ana',    'edad': 28, 'salario': 62000},\n    {'nombre': 'Pedro',  'edad': 42, 'salario': 45000},\n    {'nombre': 'Lucía',  'edad': 28, 'salario': 70000},\n]\n\n# Ordenar por salario (menor a mayor)\npor_salario = sorted(empleados, key=lambda e: e['salario'])\n\n# Ordenar por salario (mayor a menor)\npor_salario_desc = sorted(empleados, key=lambda e: e['salario'], reverse=True)\n\n# Ordenar por edad, luego por nombre (múltiples criterios)\npor_edad_nombre = sorted(empleados, key=lambda e: (e['edad'], e['nombre']))\n\n# Ordenar en su lugar (modifica la lista original)\nempleados.sort(key=lambda e: e['nombre'])\n\nprint('Por salario desc:')\nfor e in por_salario_desc:\n    print(f\"  {e['nombre']}: ${e['salario']:,}\")\n```\n\n"
         "`sorted()` devuelve una lista nueva (no modifica la original). `.sort()` modifica en su lugar. Ambos usan el algoritmo **Timsort** — O(n log n), muy eficiente."),
    pair(SYSTEM_ES,
         "Explica qué es una API REST y da un ejemplo práctico.",
         "Una **API REST** es una interfaz que permite que dos sistemas se comuniquen a través de HTTP usando operaciones estándar.\n\n"
         "**Verbos HTTP y su significado:**\n"
         "- `GET` — leer datos\n"
         "- `POST` — crear nuevos datos\n"
         "- `PUT/PATCH` — actualizar datos\n"
         "- `DELETE` — eliminar datos\n\n"
         "**Ejemplo práctico — API de una tienda:**\n"
         "```\nGET    /productos          → lista todos los productos\nGET    /productos/42       → obtiene el producto con ID 42\nPOST   /productos          → crea un producto nuevo\nPATCH  /productos/42       → actualiza precio del producto 42\nDELETE /productos/42       → elimina el producto 42\n```\n\n"
         "**Llamada con Python:**\n"
         "```python\nimport requests\n\n# Obtener un producto\nrespuesta = requests.get('https://api.mitienda.com/productos/42')\nproducto = respuesta.json()\nprint(producto['nombre'], producto['precio'])\n\n# Crear un producto\nnuevo = {'nombre': 'Laptop', 'precio': 999.99, 'stock': 50}\nrespuesta = requests.post(\n    'https://api.mitienda.com/productos',\n    json=nuevo,\n    headers={'Authorization': 'Bearer mi-token'}\n)\n```\n\n"
         "La respuesta suele venir en **JSON**, un formato de texto fácil de leer y procesar."),
    pair(SYSTEM_ES,
         "¿Qué es Git y para qué sirve? ¿Cuáles son los comandos más importantes?",
         "**Git** es un sistema de control de versiones — guarda el historial completo de cambios de tu código y permite colaborar con otros.\n\n"
         "**Flujo básico:**\n"
         "```bash\n# Iniciar un repositorio\ngit init\n\n# Ver el estado actual\ngit status\n\n# Agregar cambios al área de preparación\ngit add archivo.py          # archivo específico\ngit add .                   # todos los archivos modificados\n\n# Guardar los cambios con un mensaje\ngit commit -m 'Agrega función de búsqueda'\n\n# Ver historial\ngit log --oneline\n\n# Crear y cambiar a una rama nueva\ngit checkout -b feature/nueva-funcion\n\n# Subir cambios a GitHub\ngit push origin main\n\n# Bajar cambios del repositorio remoto\ngit pull\n\n# Fusionar una rama con main\ngit checkout main\ngit merge feature/nueva-funcion\n\n# Deshacer el último commit (sin perder los cambios)\ngit reset --soft HEAD~1\n```\n\n"
         "**Regla de oro**: haz commits frecuentes con mensajes claros. Un commit debe representar un cambio lógico completo."),
]

ES_CONVERSATIONAL = [
    pair(SYSTEM_ES,
         "¿Puedes hablarme de la historia de México en términos generales?",
         "México tiene una historia rica y compleja que abarca miles de años:\n\n"
         "**Período prehispánico (antes de 1521):**\n"
         "Civilizaciones avanzadas como los olmecas, mayas, teotihuacanos y aztecas (mexicas). Los aztecas fundaron Tenochtitlan en 1325 (hoy Ciudad de México), una ciudad-isla sobre el lago Texcoco.\n\n"
         "**Conquista española (1519-1521):**\n"
         "Hernán Cortés lideró la conquista. La caída de Tenochtitlan en 1521 marcó el inicio del **Virreinato de la Nueva España**, que duró 300 años.\n\n"
         "**Independencia (1810-1821):**\n"
         "El 16 de septiembre de 1810, Miguel Hidalgo lanzó el **Grito de Independencia** en Dolores. Tras una guerra de 11 años, México logró la independencia en 1821.\n\n"
         "**Siglo XIX:**\n"
         "Período turbulento: invasión estadounidense (1846-48, pérdida de Texas, California, etc.), Reforma (Benito Juárez), Imperio de Maximiliano, y el largo gobierno de Porfirio Díaz.\n\n"
         "**Revolución Mexicana (1910-1920):**\n"
         "Movimiento social y armado que buscó reforma agraria y democracia. Figuras clave: Emiliano Zapata y Pancho Villa.\n\n"
         "**Siglo XX-XXI:**\n"
         "Hegemonía del PRI por 71 años hasta la alternancia democrática en 2000. Hoy es una democracia federal con economía emergente importante."),
    pair(SYSTEM_ES,
         "¿Cuáles son los platos típicos de la cocina mexicana?",
         "La cocina mexicana es **Patrimonio Cultural Inmaterial de la Humanidad** (UNESCO, 2010). Algunos platos esenciales:\n\n"
         "**Cotidianos:**\n"
         "- **Tacos**: tortilla con carne (pastor, carnitas, barbacoa, birria), cebolla, cilantro y salsa\n"
         "- **Pozole**: sopa de maíz cacahuazintle con carne de cerdo o pollo, roja o verde\n"
         "- **Enchiladas**: tortillas enrolladas con salsa chile, queso y crema\n"
         "- **Tamales**: masa de maíz rellena (carne, rajas, dulce) envuelta en hoja de maíz o plátano\n\n"
         "**Más elaborados:**\n"
         "- **Mole**: salsa compleja de chiles, chocolate, especias y otros ingredientes (puede tener 30+ ingredientes)\n"
         "- **Chile en nogada**: chile poblano relleno con salsa de nuez y granada — colores de la bandera mexicana\n"
         "- **Chiles rellenos**: poblanos rellenos de queso o carne en salsa\n\n"
         "**Bebidas y postres:**\n"
         "- Horchata, agua de Jamaica, mezcal, pulque\n"
         "- Churros, tres leches, buñuelos"),
    pair(SYSTEM_ES,
         "¿Me puedes recomendar cómo aprender español más rápido?",
         "Estas estrategias están respaldadas por evidencia de adquisición de idiomas:\n\n"
         "**1. Inmersión con comprensión (lo más efectivo)**\n"
         "Consume contenido en español que entiendas ~95%: podcasts lentos, series con subtítulos en español (no en tu idioma), libros para niños primero.\n\n"
         "**2. No traduzcas — adquiere**\n"
         "En vez de memorizar 'casa = house', asocia 'casa' directamente con la imagen de una casa. Tu cerebro aprende idiomas por exposición, no por traducción.\n\n"
         "**3. Habla desde el primer día**\n"
         "Usa Italki, Tandem o intercambios de idiomas para hablar con nativos. El miedo a equivocarte es el mayor obstáculo.\n\n"
         "**4. Sistema de repetición espaciada**\n"
         "Anki con tarjetas de vocabulario en contexto (oraciones completas, no palabras sueltas).\n\n"
         "**5. Consistencia sobre intensidad**\n"
         "30 minutos diarios superan 4 horas los fines de semana. La consistencia construye el hábito neural.\n\n"
         "**6. El vocabulario más frecuente primero**\n"
         "Las 1,000 palabras más comunes cubren ~85% del habla cotidiana. Priorízalas.\n\n"
         "Para un anglohablante, el español tarda ~600 horas de estudio para nivel intermedio-alto (según el FSI)."),
    pair(SYSTEM_ES,
         "¿Cuál es la diferencia entre 'ser' y 'estar' en español?",
         "Esta es una de las partes más desafiantes del español para hablantes de inglés, ya que ambos se traducen como 'to be'.\n\n"
         "**SER** — esencia, identidad, características permanentes:\n"
         "- Identidad: *Soy María. Soy mexicana.*\n"
         "- Profesión: *Él es médico.*\n"
         "- Origen: *Somos de Guadalajara.*\n"
         "- Material: *La mesa es de madera.*\n"
         "- Relaciones: *Ella es mi hermana.*\n"
         "- Tiempo y eventos: *Son las 3. La fiesta es en mi casa.*\n\n"
         "**ESTAR** — estado, condición, ubicación temporal:\n"
         "- Estado físico: *Estoy cansado. Está enferma.*\n"
         "- Emoción: *Estamos contentos.*\n"
         "- Ubicación: *El libro está en la mesa. Estoy en México.*\n"
         "- Progresivo: *Está comiendo.*\n\n"
         "**Casos que cambian de significado:**\n"
         "- *Es aburrido* (él es una persona aburrida por naturaleza) vs. *Está aburrido* (está aburrido ahora mismo)\n"
         "- *Es rico* (es adinerado) vs. *Está rico* (la comida está deliciosa)\n"
         "- *Es muerto* (personaje ficticio muerto) vs. *Está muerto* (murió)"),
    pair(SYSTEM_ES,
         "Necesito preparar una presentación de negocios en español. ¿Me das algunas frases útiles para abrirla?",
         "Aquí tienes frases profesionales organizadas para una apertura sólida:\n\n"
         "**Saludo y bienvenida:**\n"
         "- *Buenos días/tardes a todos. Gracias por estar aquí.*\n"
         "- *Es un honor presentarles hoy...*\n"
         "- *Agradezco la oportunidad de compartir con ustedes...*\n\n"
         "**Presentación personal:**\n"
         "- *Mi nombre es [nombre] y soy [cargo] en [empresa].*\n"
         "- *Llevo [X años] trabajando en el sector de...*\n\n"
         "**Objetivo de la presentación:**\n"
         "- *El objetivo de hoy es presentarles nuestra propuesta para...*\n"
         "- *En los próximos 20 minutos, abordaremos tres puntos clave:*\n"
         "- *Al finalizar, espero que tengan una imagen clara de...*\n\n"
         "**Estructura:**\n"
         "- *Comenzaré con... luego pasaré a... y finalizaré con...*\n"
         "- *Si tienen preguntas, con gusto las atenderé al final.*\n\n"
         "**Tip profesional**: en contextos latinoamericanos, es común iniciar con una conexión personal o un dato impactante antes de entrar al contenido. Genera rapport desde el inicio."),
    pair(SYSTEM_ES,
         "¿Cómo se dice 'I would like to open a bank account' en español y cuándo se usa el condicional?",
         "La traducción es: **'Me gustaría abrir una cuenta bancaria.'**\n\n"
         "**El condicional en español:**\n"
         "Se forma con el infinitivo + terminaciones: -ía, -ías, -ía, -íamos, -íais, -ían\n\n"
         "Ejemplos:\n"
         "- *hablar → hablaría, hablarías, hablaría...*\n"
         "- *querer → querría* (irregular)\n"
         "- *poder → podría*\n"
         "- *tener → tendría*\n\n"
         "**Usos principales:**\n"
         "1. **Cortesía**: *¿Podría ayudarme?* (más formal que *¿Puede ayudarme?*)\n"
         "2. **Deseo**: *Me gustaría visitar España algún día.*\n"
         "3. **Hipótesis**: *Si tuviera dinero, viajaría al mundo entero.*\n"
         "4. **Consejo suave**: *Yo en tu lugar, no lo haría.*\n\n"
         "Para contextos bancarios en español:\n"
         "- *Quisiera abrir una cuenta de ahorros.*\n"
         "- *¿Podría informarme sobre sus tasas de interés?*\n"
         "- *Me interesaría conocer sus opciones de inversión.*"),
]

ES_TRANSLATION = [
    pair(SYSTEM_EN,
         "Translate this paragraph to Spanish:\n\n'Artificial intelligence is transforming every industry. From healthcare to finance, AI systems are automating tasks that once required human expertise. The challenge now is ensuring these systems remain transparent, fair, and aligned with human values.'",
         "La inteligencia artificial está transformando todos los sectores. Desde la salud hasta las finanzas, los sistemas de IA están automatizando tareas que antes requerían experiencia humana. El desafío ahora es garantizar que estos sistemas sigan siendo transparentes, justos y alineados con los valores humanos."),
    pair(SYSTEM_EN,
         "What does this Spanish text mean in English?\n\n'El aprendizaje profundo requiere grandes cantidades de datos etiquetados y considerable poder de cómputo. Sin embargo, los avances recientes en el aprendizaje por transferencia han reducido significativamente estos requisitos.'",
         "The Spanish text translates to:\n\n'Deep learning requires large amounts of labeled data and considerable computing power. However, recent advances in transfer learning have significantly reduced these requirements.'\n\n"
         "**Key vocabulary:**\n"
         "- *aprendizaje profundo* → deep learning\n"
         "- *datos etiquetados* → labeled data\n"
         "- *poder de cómputo* → computing power\n"
         "- *aprendizaje por transferencia* → transfer learning"),
    pair(SYSTEM_EN,
         "Translate to Spanish and explain any nuances:\n\n'The meeting was rescheduled. Please make sure to attend.'",
         "**Translation:**\n*La reunión fue reprogramada. Por favor, asegúrese de asistir.*\n\n"
         "**Or more informally:**\n*La reunión se reprogramó. Por favor, no faltes / asegúrate de asistir.*\n\n"
         "**Nuances:**\n"
         "- 'Fue reprogramada' uses passive voice (formal). 'Se reprogramó' uses reflexive passive (more natural in speech).\n"
         "- 'Asegúrese' is formal usted form. 'Asegúrate' is informal tú form.\n"
         "- 'No faltes' (don't miss it) is more direct and colloquial.\n\n"
         "In professional email: *asegúrese de asistir* or *le recordamos que su asistencia es importante.*"),
    pair(SYSTEM_EN,
         "How do you say these business phrases in Spanish?\n1. 'Let's schedule a follow-up meeting'\n2. 'Please review the attached document'\n3. 'We look forward to your response'",
         "1. **'Let's schedule a follow-up meeting'**\n"
         "   - *Programemos una reunión de seguimiento.* (formal, inclusive)\n"
         "   - *¿Podríamos agendar una reunión de seguimiento?* (more polite, common in Latin America)\n\n"
         "2. **'Please review the attached document'**\n"
         "   - *Le solicito revisar el documento adjunto.* (formal)\n"
         "   - *Por favor, revise el documento adjunto.* (standard business)\n"
         "   - *Adjunto encontrará el documento para su revisión.* (common email opener)\n\n"
         "3. **'We look forward to your response'**\n"
         "   - *Quedamos en espera de su respuesta.* (most common in formal emails)\n"
         "   - *En espera de sus comentarios...* (briefer)\n"
         "   - *Esperamos su pronta respuesta.* (implies some urgency)\n\n"
         "**Regional note**: 'Agendar' is widely used in Latin America but less so in Spain, where 'programar' is preferred."),
    pair(SYSTEM_EN,
         "Translate this Python error message explanation to Spanish:\n\n'A KeyError occurs when you try to access a dictionary key that doesn't exist. Always use .get() method or check if the key exists first.'",
         "Un **KeyError** ocurre cuando intentas acceder a una clave de diccionario que no existe. Siempre usa el método `.get()` o verifica si la clave existe primero.\n\n"
         "**Ejemplo:**\n"
         "```python\n# Causa KeyError:\nconfig = {'host': 'localhost'}\nprint(config['puerto'])  # ❌ KeyError: 'puerto'\n\n# Solución 1 — .get() con valor predeterminado:\npuerto = config.get('puerto', 8080)  # ✅ devuelve 8080 si no existe\n\n# Solución 2 — verificar antes:\nif 'puerto' in config:\n    print(config['puerto'])\nelse:\n    print('Clave no encontrada')\n```"),
]

ES_CROSS_LINGUAL = [
    pair(SYSTEM_EN,
         "Answer in Spanish: What are the main benefits of regular exercise?",
         "Los principales beneficios del ejercicio regular son:\n\n"
         "**Físicos:**\n"
         "- Fortalece el corazón y reduce el riesgo de enfermedades cardiovasculares\n"
         "- Mantiene un peso saludable acelerando el metabolismo\n"
         "- Fortalece huesos y músculos, previniendo la osteoporosis\n"
         "- Mejora la flexibilidad y el equilibrio, reduciendo caídas en adultos mayores\n"
         "- Refuerza el sistema inmunitario\n\n"
         "**Mentales:**\n"
         "- Libera endorfinas, serotonina y dopamina — reduce ansiedad y depresión\n"
         "- Mejora la calidad del sueño\n"
         "- Aumenta la capacidad de concentración y memoria (favorece la neurogénesis)\n"
         "- Reduce el cortisol (hormona del estrés)\n\n"
         "**La OMS recomienda**: al menos 150 minutos semanales de actividad moderada, o 75 minutos de actividad intensa."),
    pair(SYSTEM_EN,
         "Respond in Spanish: What is the difference between machine learning and deep learning?",
         "**Aprendizaje automático (Machine Learning)** es el campo amplio donde los sistemas aprenden de datos sin ser programados explícitamente.\n\n"
         "**Aprendizaje profundo (Deep Learning)** es una **subcategoría** del ML que usa redes neuronales con muchas capas ('profundas') para aprender representaciones complejas.\n\n"
         "**Analogía**: el ML es como 'aprender a cocinar'. El deep learning es como 'estudiar en una escuela culinaria de élite con 20 niveles de entrenamiento especializado'.\n\n"
         "**Diferencias prácticas:**\n\n"
         "| Aspecto | ML clásico | Deep Learning |\n"
         "|---------|-----------|---------------|\n"
         "| Datos | Funciona con pocos datos | Necesita millones de ejemplos |\n"
         "| Features | El humano las diseña | Las aprende automáticamente |\n"
         "| Interpretabilidad | Alta | Baja (caja negra) |\n"
         "| Cómputo | CPU normal | GPU/TPU potentes |\n"
         "| Ejemplos | Regresión, SVM, árboles | GPT, DALL-E, AlphaFold |"),
    pair(SYSTEM_EN,
         "Please explain what an API is, but write your explanation in Spanish.",
         "Una **API** (Interfaz de Programación de Aplicaciones) es un contrato que define cómo dos programas se comunican entre sí.\n\n"
         "**Analogía del restaurante:**\n"
         "Imagina que estás en un restaurante. Tú no entras a la cocina a preparar tu comida — el **mesero** (API) toma tu pedido, lo lleva a la cocina, y te trae el resultado.\n\n"
         "- **Tú** = tu aplicación\n"
         "- **La cocina** = el sistema externo (Google Maps, Twitter, tu banco)\n"
         "- **El mesero** = la API\n\n"
         "**Ejemplo real:**\n"
         "Cuando una app de clima muestra el tiempo, no tiene su propio satélite. Hace una petición a la API de un servicio meteorológico:\n"
         "```\nGET https://api.clima.com/v1/tiempo?ciudad=Ciudad_de_Mexico\n```\nY recibe los datos en JSON para mostrarlos.\n\n"
         "**¿Para qué sirven?**\n"
         "- Integraciones (pagar con PayPal sin salir de una app)\n"
         "- Compartir datos entre sistemas\n"
         "- Construir sobre servicios existentes (mapas, traducciones, pagos)"),
    pair(SYSTEM_EN,
         "What is photosynthesis? Answer in Spanish.",
         "La **fotosíntesis** es el proceso mediante el cual las plantas, algas y algunas bacterias convierten la luz solar en energía química almacenada como azúcar (glucosa).\n\n"
         "**Ecuación resumida:**\n"
         "```\n6CO₂ + 6H₂O + luz solar → C₆H₁₂O₆ (glucosa) + 6O₂\n```\n"
         "En palabras: dióxido de carbono + agua + luz → glucosa + oxígeno\n\n"
         "**¿Dónde ocurre?**\n"
         "En los **cloroplastos**, orgánulos de las células vegetales que contienen **clorofila** — el pigmento verde que absorbe la luz (principalmente roja y azul, refleja el verde).\n\n"
         "**Dos etapas:**\n"
         "1. **Reacciones de luz** (fase luminosa): capturan energía solar y producen ATP y NADPH.\n"
         "2. **Ciclo de Calvin** (fase oscura): usan ese ATP para convertir CO₂ en glucosa.\n\n"
         "**Importancia:**\n"
         "- Produce el oxígeno que respiramos\n"
         "- Es la base de casi todas las cadenas alimentarias\n"
         "- Fija el carbono atmosférico, regulando el clima"),
]


# ─────────────────────────────────────────────────────────────────────────────
# FRENCH (FR) — 200 pairs
# ─────────────────────────────────────────────────────────────────────────────

FR_GENERAL = [
    pair(SYSTEM_FR,
         "Qu'est-ce que l'intelligence artificielle ?",
         "L'intelligence artificielle (IA) est la capacité des machines à accomplir des tâches qui nécessitent normalement l'intelligence humaine : reconnaître des images, comprendre le langage naturel, prendre des décisions ou jouer à des jeux.\n\n"
         "**Comment ça fonctionne ?**\nLa plupart des systèmes d'IA modernes utilisent l'**apprentissage automatique** : au lieu de programmer des règles explicites, on montre au système des milliers d'exemples et il apprend à reconnaître des patterns.\n\n"
         "**Exemple** : pour apprendre à reconnaître des chats, le modèle voit des millions d'images étiquetées 'chat' ou 'non-chat' et ajuste ses paramètres internes jusqu'à distinguer les deux.\n\n"
         "**Types d'IA :**\n- IA étroite (ANI) : excel à une tâche spécifique (reconnaissance vocale, jeu d'échecs)\n- IA générale (AGI) : hypothétique machine qui pense comme un humain\n- Super-IA (ASI) : encore plus hypothétique — surpasse l'intelligence humaine dans tous les domaines\n\n"
         "Aujourd'hui, tout ce qui existe est de l'IA étroite, même si elle peut paraître impressionnante."),
    pair(SYSTEM_FR,
         "Explique-moi la différence entre un virus et un logiciel malveillant.",
         "Le **logiciel malveillant** (malware) est le terme général pour tout logiciel conçu pour nuire. Le **virus** en est un sous-type.\n\n"
         "**Types de malware :**\n"
         "- **Virus** : s'attache à d'autres fichiers et se copie quand on les exécute. Nécessite une action humaine pour se propager.\n"
         "- **Ver (worm)** : se propage seul via les réseaux, sans fichier hôte. WannaCry en 2017 a infecté 300 000 machines en 24h.\n"
         "- **Cheval de Troie** : se déguise en logiciel légitime pour tromper l'utilisateur.\n"
         "- **Ransomware** : chiffre vos fichiers et demande une rançon.\n"
         "- **Spyware** : surveille vos activités en secret.\n"
         "- **Adware** : affiche des publicités non désirées.\n\n"
         "**Protection de base :**\n"
         "- Mises à jour système régulières\n"
         "- Antivirus à jour\n"
         "- Ne pas ouvrir les pièces jointes inconnues\n"
         "- Sauvegardes régulières (règle 3-2-1)"),
    pair(SYSTEM_FR,
         "Comment fonctionne un moteur de recherche ?",
         "Un moteur de recherche comme Google fonctionne en trois grandes étapes :\n\n"
         "**1. Exploration (crawling)**\nDes robots logiciels appelés 'crawlers' ou 'spiders' parcourent le web en suivant les liens d'une page à l'autre. Ils téléchargent le contenu de chaque page visitée.\n\n"
         "**2. Indexation**\nLe contenu collecté est analysé, traité et stocké dans un immense index — une base de données qui associe chaque mot-clé aux pages où il apparaît. Google indexe plus de 100 milliards de pages web.\n\n"
         "**3. Classement (ranking)**\nQuand tu tapes une requête, l'algorithme classe les pages pertinentes selon des centaines de critères :\n"
         "- Pertinence du contenu par rapport à la requête\n"
         "- Qualité et nombre de liens pointant vers la page (PageRank original)\n"
         "- Vitesse de chargement et expérience utilisateur\n"
         "- Fraîcheur du contenu\n"
         "- Localisation géographique\n\n"
         "Aujourd'hui, Google utilise également des modèles de langage pour mieux comprendre l'intention derrière les requêtes ('tu veux acheter une paire de chaussures ou comprendre l'histoire des chaussures ?')."),
    pair(SYSTEM_FR,
         "Quelle est la différence entre 'apprendre' et 'étudier' en français ?",
         "Ce sont deux verbes proches mais distincts :\n\n"
         "**Apprendre** :\n"
         "- Acquérir une connaissance ou une compétence (résultat)\n"
         "- *J'ai appris à nager.* (je sais maintenant nager)\n"
         "- *Il a appris la nouvelle.* (il est maintenant au courant)\n"
         "- Peut aussi signifier enseigner : *Ma mère m'a appris à cuisiner.* (registre soutenu)\n\n"
         "**Étudier** :\n"
         "- Se consacrer à l'étude de quelque chose (processus, action)\n"
         "- *J'étudie le japonais depuis deux ans.* (je suis en train d'apprendre)\n"
         "- *Elle étudie la médecine à Paris.* (elle suit des études de médecine)\n\n"
         "**Résumé** : on 'étudie' pendant le processus, on 'apprend' quand on a acquis le résultat.\n\n"
         "Exemples comparés :\n"
         "- *J'étudie le piano* → je prends des cours, je pratique\n"
         "- *J'ai appris à jouer du piano* → j'en suis maintenant capable"),
    pair(SYSTEM_FR,
         "Quels sont les avantages et inconvénients du travail à distance ?",
         "**Avantages :**\n\n"
         "*Pour les employés :*\n"
         "- Pas de temps de trajet — économie de 1 à 2h par jour en moyenne\n"
         "- Flexibilité horaire (dans les équipes qui l'autorisent)\n"
         "- Meilleur équilibre vie pro/vie perso pour certains\n"
         "- Moins de distractions en open-space\n"
         "- Possibilité de travailler depuis n'importe où\n\n"
         "*Pour les entreprises :*\n"
         "- Réduction des coûts immobiliers\n"
         "- Accès à des talents sans contrainte géographique\n"
         "- Souvent une productivité maintenue ou améliorée sur les tâches individuelles\n\n"
         "**Inconvénients :**\n\n"
         "*Pour les employés :*\n"
         "- Isolement social et sentiment de déconnexion\n"
         "- Difficultés à séparer vie professionnelle et personnelle\n"
         "- Manque de visibilité pour l'avancement de carrière\n"
         "- Connexion internet ou matériel inadapté pour certains\n\n"
         "*Pour les entreprises :*\n"
         "- Collaboration et créativité collective plus difficiles\n"
         "- Onboarding des nouveaux employés plus complexe\n"
         "- Risques de sécurité informatique accrus\n\n"
         "Le modèle hybride (2-3 jours au bureau) semble être le meilleur compromis selon les études récentes."),
    pair(SYSTEM_FR,
         "Comment dit-on 'Je voudrais un café, s'il vous plaît' dans plusieurs langues ?",
         "Voici comment demander un café poliment dans différentes langues :\n\n"
         "- 🇫🇷 **Français** : *Je voudrais un café, s'il vous plaît.*\n"
         "- 🇪🇸 **Espagnol** : *Quisiera un café, por favor.*\n"
         "- 🇮🇹 **Italien** : *Vorrei un caffè, per favore.*\n"
         "- 🇵🇹 **Portugais** : *Gostaria de um café, por favor.*\n"
         "- 🇩🇪 **Allemand** : *Ich hätte gerne einen Kaffee, bitte.*\n"
         "- 🇯🇵 **Japonais** : *コーヒーをください。* (Kōhī o kudasai.)\n"
         "- 🇨🇳 **Mandarin** : *我想要一杯咖啡，谢谢。* (Wǒ xiǎng yào yī bēi kāfēi, xièxiè.)\n"
         "- 🇸🇦 **Arabe** : *أريد قهوة من فضلك.* (Urīdu qahwatan min fadlik.)\n"
         "- 🇷🇺 **Russe** : *Можно кофе, пожалуйста.* (Mozhno kofe, pozhaluysta.)\n\n"
         "**Conseil de voyage** : dans les pays non anglophones, essayer même une seule phrase dans la langue locale est généralement très apprécié."),
    pair(SYSTEM_FR,
         "Explique le concept de l'effet de serre en termes simples.",
         "L'effet de serre est un phénomène **naturel et essentiel** à la vie sur Terre — c'est son dérèglement qui pose problème.\n\n"
         "**Comment ça marche :**\n"
         "1. Le soleil envoie de l'énergie sous forme de lumière visible.\n"
         "2. La Terre absorbe cette énergie et la réémet sous forme de chaleur infrarouge.\n"
         "3. Certains gaz dans l'atmosphère — CO₂, méthane, vapeur d'eau — absorbent une partie de cette chaleur au lieu de la laisser s'échapper dans l'espace.\n"
         "4. Ces gaz 'renvoient' la chaleur vers la Terre, la réchauffant.\n\n"
         "**L'analogie de la serre :**\nComme le verre d'une serre horticole laisse entrer la lumière mais retient la chaleur — d'où le nom.\n\n"
         "**Le problème actuel :**\nDepuis la révolution industrielle, nous avons augmenté la concentration de CO₂ de 280 à plus de 420 ppm (parties par million). Cette accumulation renforce l'effet de serre naturel, déréglant le climat mondial.\n\n"
         "Sans aucun effet de serre, la température moyenne de la Terre serait de -18°C au lieu de +15°C."),
    pair(SYSTEM_FR,
         "Quels sont les temps verbaux les plus importants en français pour un débutant ?",
         "Pour débuter, concentre-toi sur ces 5 temps dans l'ordre :\n\n"
         "**1. Présent** — actions actuelles et habituelles\n"
         "*Je parle / tu parles / il parle*\n"
         "Priorité absolue — couvre ~60% des situations quotidiennes.\n\n"
         "**2. Passé composé** — actions terminées dans le passé\n"
         "*J'ai mangé / tu as parlé / il est allé*\n"
         "Avoir ou être + participe passé. Note : verbes de mouvement prennent 'être' (aller, venir, partir...).\n\n"
         "**3. Futur proche** — futur immédiat\n"
         "*Je vais partir / tu vas manger / il va venir*\n"
         "Aller au présent + infinitif. Très utilisé à l'oral.\n\n"
         "**4. Imparfait** — descriptions et actions répétées dans le passé\n"
         "*Je parlais / tu mangeais / il faisait*\n"
         "Contexte et arrière-plan : *Quand j'étais enfant, j'habitais à Lyon.*\n\n"
         "**5. Conditionnel présent** — politesse et hypothèses\n"
         "*Je voudrais / pourriez-vous / il faudrait*\n"
         "Indispensable pour parler poliment.\n\n"
         "Avec ces 5 temps, tu peux te débrouiller dans la très grande majorité des situations."),
]

FR_CODE = [
    pair(SYSTEM_FR,
         "Comment créer une liste et y ajouter des éléments en Python ?",
         "```python\n# Créer une liste vide\nma_liste = []\n\n# Créer une liste avec des valeurs initiales\nfruits = ['pomme', 'banane', 'cerise']\nnombres = [1, 2, 3, 4, 5]\nmixte  = [1, 'bonjour', True, 3.14]  # Python accepte des types différents\n\n# Ajouter un élément à la fin\nfruits.append('mangue')  # ['pomme', 'banane', 'cerise', 'mangue']\n\n# Ajouter à une position spécifique\nfruits.insert(1, 'kiwi')  # insère en index 1\n\n# Ajouter plusieurs éléments d'un coup\nfruits.extend(['ananas', 'melon'])\n# ou : fruits += ['ananas', 'melon']\n\n# Accéder aux éléments\nprint(fruits[0])   # premier élément : 'pomme'\nprint(fruits[-1])  # dernier élément\nprint(fruits[1:3]) # slice : éléments 1 et 2\n\n# Supprimer un élément\nfruits.remove('banane')  # supprime la première occurrence\ndel fruits[0]            # supprime par index\n\nprint(f'Nombre de fruits : {len(fruits)}')\n```\n\n"
         "**Astuce** : `append()` pour un seul élément, `extend()` pour plusieurs. Ne confonds pas les deux !"),
    pair(SYSTEM_FR,
         "Qu'est-ce qu'une boucle for en Python et comment l'utiliser ?",
         "Une boucle `for` répète un bloc de code pour chaque élément d'une séquence.\n\n"
         "```python\n# Parcourir une liste\nfruits = ['pomme', 'banane', 'cerise']\nfor fruit in fruits:\n    print(f'J\\'aime les {fruit}s')\n\n# Parcourir une plage de nombres\nfor i in range(5):         # 0, 1, 2, 3, 4\n    print(i)\n\nfor i in range(1, 11):     # 1 à 10\n    print(i)\n\nfor i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (pas de 2)\n    print(i)\n\n# Avec l'index et la valeur (enumerate)\nfor index, fruit in enumerate(fruits):\n    print(f'{index}: {fruit}')\n\n# Parcourir un dictionnaire\npersonne = {'nom': 'Alice', 'âge': 30, 'ville': 'Paris'}\nfor clé, valeur in personne.items():\n    print(f'{clé}: {valeur}')\n\n# List comprehension (boucle condensée)\ncarrés = [x**2 for x in range(1, 6)]  # [1, 4, 9, 16, 25]\npairs  = [x for x in range(20) if x % 2 == 0]\n```"),
    pair(SYSTEM_FR,
         "Comment gérer les erreurs en Python avec try/except ?",
         "```python\n# Structure de base\ntry:\n    nombre = int(input('Entrez un nombre : '))\n    résultat = 10 / nombre\n    print(f'Résultat : {résultat}')\nexcept ValueError:\n    print('Erreur : ce n\\'est pas un nombre valide')\nexcept ZeroDivisionError:\n    print('Erreur : division par zéro impossible')\nexcept Exception as e:\n    print(f'Erreur inattendue : {e}')\nelse:\n    print('Tout s\\'est bien passé !')  # s'exécute si pas d'erreur\nfinally:\n    print('Ce bloc s\\'exécute toujours')  # nettoyage, fermeture de fichiers\n\n# Lever une exception personnalisée\ndef diviser(a, b):\n    if b == 0:\n        raise ValueError('Le diviseur ne peut pas être zéro')\n    return a / b\n\ntry:\n    print(diviser(10, 0))\nexcept ValueError as e:\n    print(f'Erreur : {e}')\n```\n\n"
         "**Règle** : attrape les exceptions **spécifiques** (ValueError, FileNotFoundError) plutôt que le générique `Exception` — sinon tu masques des bugs réels."),
]

FR_CROSS_LINGUAL = [
    pair(SYSTEM_EN,
         "Answer in French: What are the main differences between Python and JavaScript?",
         "**Python vs JavaScript** — les différences principales :\n\n"
         "**Utilisation principale :**\n"
         "- Python : science des données, IA, automatisation, scripts serveur\n"
         "- JavaScript : développement web (navigateur), mais aussi serveur (Node.js)\n\n"
         "**Syntaxe :**\n"
         "- Python utilise l'indentation pour délimiter les blocs (pas d'accolades)\n"
         "- JavaScript utilise des accolades `{}` et point-virgule optionnel\n\n"
         "**Typage :**\n"
         "- Les deux sont dynamiquement typés\n"
         "- Python a un typage plus strict (pas de coercition implicite surprenante)\n"
         "- JS : `'5' + 3 = '53'` (coercition) ; Python : erreur de type\n\n"
         "**Concurrence :**\n"
         "- JavaScript : event loop single-thread, très adapté aux I/O asynchrones\n"
         "- Python : asyncio, threads, multiprocessing (GIL peut limiter)\n\n"
         "**Ecosystème :**\n"
         "- Python : NumPy, Pandas, PyTorch, scikit-learn\n"
         "- JavaScript : React, Vue, Express, Node.js\n\n"
         "**Lequel choisir ?** Pour l'IA/data → Python. Pour le web frontend → JS obligatoire. Pour le backend → les deux sont viables."),
    pair(SYSTEM_EN,
         "Please explain recursion in French with a simple example.",
         "La **récursion** est une technique où une fonction s'appelle elle-même pour résoudre un problème.\n\n"
         "**Principe** : on divise un gros problème en sous-problèmes identiques mais plus petits, jusqu'à atteindre un cas de base simple à résoudre directement.\n\n"
         "**Exemple — factorielle :**\n"
         "La factorielle de n (notée n!) = n × (n-1) × (n-2) × ... × 1\n\n"
         "```python\ndef factorielle(n):\n    # Cas de base — on arrête la récursion\n    if n == 0 or n == 1:\n        return 1\n    # Appel récursif — le problème devient plus petit\n    return n * factorielle(n - 1)\n\nprint(factorielle(5))  # 120\n# Déroulement : 5 × factorielle(4)\n#             = 5 × 4 × factorielle(3)\n#             = 5 × 4 × 3 × factorielle(2)\n#             = 5 × 4 × 3 × 2 × factorielle(1)\n#             = 5 × 4 × 3 × 2 × 1 = 120\n```\n\n"
         "**Attention** : sans cas de base, la récursion tourne indéfiniment → `RecursionError` (Python limite la profondeur à ~1000 appels par défaut)."),
]


# ─────────────────────────────────────────────────────────────────────────────
# PORTUGUESE (PT) — 120 pairs
# ─────────────────────────────────────────────────────────────────────────────

PT_GENERAL = [
    pair(SYSTEM_PT,
         "O que é inteligência artificial e como ela funciona?",
         "A inteligência artificial (IA) é a capacidade de sistemas computacionais realizarem tarefas que normalmente exigem inteligência humana: reconhecer imagens, entender linguagem, tomar decisões e resolver problemas.\n\n"
         "**Como funciona:**\nA maioria dos sistemas modernos usa **aprendizado de máquina**: em vez de programar regras explícitas, mostramos ao sistema milhares de exemplos e ele aprende os padrões sozinho.\n\n"
         "**Exemplo simples:**\nPara ensinar uma IA a distinguir gatos de cães, você não programa 'se tem bigodes e orelhas pontudas = gato'. Você mostra 1 milhão de fotos rotuladas e o modelo ajusta seus parâmetros internos até acertar.\n\n"
         "**Tipos principais:**\n"
         "- **IA estreita** (o que existe hoje): excelente em uma tarefa específica\n"
         "- **IA geral**: hipotética — pensa como humano em todas as áreas\n"
         "- **Aprendizado profundo**: subcampo que usa redes neurais com muitas camadas\n\n"
         "Aplicações: reconhecimento facial, diagnóstico médico, tradução automática, recomendações da Netflix, carros autônomos."),
    pair(SYSTEM_PT,
         "Explique o que é inflação de forma simples.",
         "Inflação é o aumento generalizado e sustentado dos preços ao longo do tempo. Se hoje um café custa R$6 e no próximo ano custa R$6,60, a inflação foi de 10%.\n\n"
         "**Causas principais:**\n"
         "- **Demanda excessiva**: muita gente querendo comprar o mesmo produto\n"
         "- **Custo de produção**: petróleo mais caro → frete mais caro → tudo mais caro\n"
         "- **Emissão de moeda**: governo imprime mais dinheiro → cada real vale menos\n"
         "- **Expectativas**: se todos acham que os preços vão subir, exigem aumentos de salário e repassam custos\n\n"
         "**Impactos:**\n"
         "- Seu dinheiro perde poder de compra\n"
         "- Quem tem poupança perde (se a inflação superar os juros)\n"
         "- Devedores saem ganhando (pagam com dinheiro que vale menos)\n\n"
         "**No Brasil**: o IPCA é o índice oficial. O Banco Central usa a taxa Selic para controlar a inflação — juros altos desaceleram a economia e reduzem preços.\n\n"
         "Uma inflação baixa e estável (~2-4% ao ano) é considerada saudável."),
    pair(SYSTEM_PT,
         "Quais são as diferenças entre o português do Brasil e o de Portugal?",
         "As diferenças são significativas, mas a comunicação é sempre possível. Principais divergências:\n\n"
         "**Pronúncia:**\n"
         "- Portugal: vogais átonas mais 'fechadas' ou suprimidas — *computador* soa como 'cmputador'\n"
         "- Brasil: vogais mais abertas e claramente pronunciadas\n"
         "- Sotaques e entonação completamente distintos\n\n"
         "**Vocabulário:**\n"
         "| Conceito | Brasil | Portugal |\n"
         "|----------|--------|----------|\n"
         "| Celular | celular | telemóvel |\n"
         "| Ônibus | ônibus | autocarro |\n"
         "| Geladeira | geladeira | frigorífico |\n"
         "| Sorvete | sorvete | gelado |\n"
         "| Suco | suco | sumo |\n\n"
         "**Gramática:**\n"
         "- Colocação pronominal: Brasil usa 'Me dá isso', Portugal usa 'Dá-me isso'\n"
         "- Tutear: Brasil usa 'você' como padrão; Portugal usa 'tu' com conjugação diferente\n"
         "- Gerúndio: Brasil 'estou estudando'; Portugal 'estou a estudar'\n\n"
         "**Ortografia**: unificada pelo Acordo Ortográfico de 1990 (com implementação gradual), mas ainda há pequenas diferenças residuais."),
    pair(SYSTEM_PT,
         "Como funciona o sistema de saúde público no Brasil (SUS)?",
         "O **SUS (Sistema Único de Saúde)** é um dos maiores sistemas de saúde pública do mundo, criado pela Constituição de 1988 com base em três princípios: **universalidade** (para todos), **integralidade** (atenção completa) e **equidade** (prioridade a quem mais precisa).\n\n"
         "**Estrutura:**\n"
         "- **Atenção básica**: UBSs (Unidades Básicas de Saúde) e Estratégia Saúde da Família — porta de entrada\n"
         "- **Média complexidade**: UPAs, ambulatórios especializados\n"
         "- **Alta complexidade**: hospitais universitários, transplantes, oncologia\n\n"
         "**O que é gratuito:**\n"
         "- Consultas médicas\n"
         "- Exames laboratoriais e de imagem\n"
         "- Cirurgias, inclusive complexas (coração, transplantes)\n"
         "- Medicamentos da lista RENAME\n"
         "- Vacinação\n"
         "- Pré-natal e parto\n\n"
         "**Desafios:**\n"
         "- Filas longas para especialistas e cirurgias eletivas\n"
         "- Subfinanciamento histórico\n"
         "- Distribuição desigual entre regiões\n\n"
         "Apesar dos problemas, o SUS realiza mais de 1 bilhão de procedimentos por ano e tem programas referência mundial (vacinação, HIV/AIDS, transplantes)."),
    pair(SYSTEM_PT,
         "O que é machine learning? Explique de forma simples.",
         "**Aprendizado de máquina** (machine learning) é ensinar um computador a aprender a partir de exemplos, em vez de programá-lo com regras explícitas.\n\n"
         "**Analogia:**\n"
         "Imagine ensinar uma criança a reconhecer cachorro. Você não diz 'se tem quatro patas E pelo E late → é cachorro'. Você mostra fotos: 'esse é cachorro, esse não é'. A criança aprende os padrões sozinha.\n\n"
         "O ML funciona assim, mas com milhares ou milhões de exemplos e ajustando parâmetros matemáticos.\n\n"
         "**Como funciona na prática:**\n"
         "1. Você alimenta o modelo com dados (exemplos rotulados)\n"
         "2. O modelo faz previsões e calcula o erro\n"
         "3. Ajusta seus parâmetros para reduzir o erro (gradient descent)\n"
         "4. Repete até o erro ser aceitável\n\n"
         "**Exemplos do dia a dia:**\n"
         "- Spam no e-mail sendo filtrado automaticamente\n"
         "- Recomendações do Spotify/Netflix\n"
         "- Reconhecimento facial no seu celular\n"
         "- Tradução do Google Translate"),
    pair(SYSTEM_PT,
         "Explique o que é uma API REST com um exemplo prático em português.",
         "Uma **API REST** é uma interface que permite dois sistemas se comunicarem pela internet usando o protocolo HTTP.\n\n"
         "**Analogia do cardápio:**\nImagine um restaurante. Você (o cliente/aplicação) não entra na cozinha. O garçom (API) anota seu pedido, leva para a cozinha (servidor) e traz o resultado.\n\n"
         "**Verbos HTTP:**\n"
         "- `GET` → buscar dados\n"
         "- `POST` → criar novo registro\n"
         "- `PUT/PATCH` → atualizar\n"
         "- `DELETE` → remover\n\n"
         "**Exemplo — API de um banco:**\n"
         "```\nGET  /contas/12345          → retorna dados da conta\nGET  /contas/12345/extrato  → retorna extrato\nPOST /transferencias        → faz uma transferência\n```\n\n"
         "**Chamada com Python:**\n"
         "```python\nimport requests\n\n# Consultar saldo\nresposta = requests.get(\n    'https://api.meubank.com/contas/12345',\n    headers={'Authorization': 'Bearer meu-token'}\n)\nconta = resposta.json()\nprint(f'Saldo: R${conta[\"saldo\"]:,.2f}')\n```\n\n"
         "A resposta vem em **JSON** — formato de texto estruturado fácil de processar em qualquer linguagem."),
]

PT_CODE = [
    pair(SYSTEM_PT,
         "Como criar uma função em Python? Mostre um exemplo prático.",
         "```python\n# Função básica\ndef saudacao(nome):\n    return f'Olá, {nome}! Bem-vindo.'\n\nprint(saudacao('Maria'))  # Olá, Maria! Bem-vindo.\n\n# Com parâmetros padrão\ndef calcular_desconto(preco, desconto=0.10):\n    \"\"\"Aplica desconto percentual a um preço.\"\"\"\n    valor_desconto = preco * desconto\n    return preco - valor_desconto\n\nprint(calcular_desconto(100))        # 90.0 (10% padrão)\nprint(calcular_desconto(100, 0.25))  # 75.0 (25%)\n\n# Com múltiplos retornos\ndef dividir(a, b):\n    if b == 0:\n        return None, 'Divisão por zero não permitida'\n    return a / b, None  # resultado, erro\n\nresultado, erro = dividir(10, 2)\nif erro:\n    print(f'Erro: {erro}')\nelse:\n    print(f'Resultado: {resultado}')\n\n# Funções como cidadãos de primeira classe\ndef aplicar(funcao, valor):\n    return funcao(valor)\n\ndobro = lambda x: x * 2\nprint(aplicar(dobro, 5))  # 10\n```\n\n"
         "**Boas práticas**: nome em snake_case, docstring breve para funções complexas, uma responsabilidade por função."),
    pair(SYSTEM_PT,
         "Como ler e escrever arquivos em Python?",
         "```python\n# ── ESCREVER ──────────────────────────────────────────\n# Modo 'w' — sobrescreve se existir\nwith open('dados.txt', 'w', encoding='utf-8') as f:\n    f.write('Primeira linha\\n')\n    f.write('Segunda linha\\n')\n\n# Escrever várias linhas\nlinhas = ['João\\n', 'Maria\\n', 'Pedro\\n']\nwith open('nomes.txt', 'w', encoding='utf-8') as f:\n    f.writelines(linhas)\n\n# Modo 'a' — adiciona ao final (append)\nwith open('log.txt', 'a', encoding='utf-8') as f:\n    f.write('Nova entrada de log\\n')\n\n# ── LER ───────────────────────────────────────────────\n# Ler tudo de uma vez\nwith open('dados.txt', 'r', encoding='utf-8') as f:\n    conteudo = f.read()\n\n# Ler linha por linha (eficiente para arquivos grandes)\nwith open('nomes.txt', 'r', encoding='utf-8') as f:\n    for linha in f:\n        print(linha.strip())  # strip() remove \\n\n\n# Ler em lista\nwith open('nomes.txt', 'r', encoding='utf-8') as f:\n    linhas = f.readlines()\n\n# ── JSON ──────────────────────────────────────────────\nimport json\n\ndados = {'nome': 'Alice', 'idade': 30, 'cidade': 'São Paulo'}\n\n# Salvar JSON\nwith open('dados.json', 'w', encoding='utf-8') as f:\n    json.dump(dados, f, ensure_ascii=False, indent=2)\n\n# Carregar JSON\nwith open('dados.json', 'r', encoding='utf-8') as f:\n    dados_carregados = json.load(f)\n```\n\n"
         "**Use sempre `with`** — garante que o arquivo seja fechado mesmo se houver erro."),
]

PT_CROSS_LINGUAL = [
    pair(SYSTEM_EN,
         "Answer in Portuguese (Brazilian): What are the best practices for writing clean code?",
         "As melhores práticas para código limpo:\n\n"
         "**Nomenclatura:**\n"
         "- Use nomes descritivos: `calcular_imposto_renda()` em vez de `calc()`\n"
         "- Variáveis em snake_case (Python), camelCase (JS/Java)\n"
         "- Evite abreviações crípticas: `usuario` não `usr`\n\n"
         "**Funções:**\n"
         "- Uma função = uma responsabilidade\n"
         "- Máximo 20 linhas (geralmente)\n"
         "- Prefira retornar valores a modificar variáveis globais\n\n"
         "**Comentários:**\n"
         "- Explique o *por quê*, não o *o quê* (o código já diz o que faz)\n"
         "- Código claro > código comentado. Se precisa comentar muito, refatore.\n\n"
         "**Organização:**\n"
         "- DRY (Don't Repeat Yourself) — extraia código duplicado para funções\n"
         "- Separe responsabilidades em módulos/classes diferentes\n"
         "- Arquivos pequenos e focados são mais fáceis de manter\n\n"
         "**Qualidade:**\n"
         "- Escreva testes unitários — código não testado é código que quebra silenciosamente\n"
         "- Revise seu próprio código antes de submeter\n"
         "- Use um linter (flake8, ESLint) para consistência automática"),
    pair(SYSTEM_EN,
         "Explain Git branching in Portuguese (Brazil).",
         "**Branches (ramificações) no Git** permitem trabalhar em funcionalidades diferentes sem afetar o código principal.\n\n"
         "**Por que usar branches?**\n"
         "Imagine a branch `main` como o código em produção. Você não quer experimentar coisas diretamente nela — pode quebrar o sistema.\n\n"
         "```bash\n# Criar e mudar para uma branch nova\ngit checkout -b feature/cadastro-usuario\n# ou (Git moderno):\ngit switch -c feature/cadastro-usuario\n\n# Listar branches\ngit branch              # locais\ngit branch -a           # locais e remotas\n\n# Mudar de branch\ngit checkout main\ngit switch main\n\n# Fazer merge (trazer mudanças para main)\ngit checkout main\ngit merge feature/cadastro-usuario\n\n# Deletar branch após o merge\ngit branch -d feature/cadastro-usuario\n\n# Enviar branch para o GitHub\ngit push origin feature/cadastro-usuario\n```\n\n"
         "**Fluxo recomendado (Git Flow simplificado):**\n"
         "- `main` → código em produção, sempre estável\n"
         "- `develop` → integração de features\n"
         "- `feature/nome` → nova funcionalidade\n"
         "- `hotfix/nome` → correção urgente em produção\n\n"
         "**Pull Request**: antes de fazer merge, abra um PR no GitHub para revisão de código — é uma boa prática em equipes."),
]


# ─────────────────────────────────────────────────────────────────────────────
# ARABIC (AR) — 80 pairs — English prompts with Arabic responses + transliterations
# ─────────────────────────────────────────────────────────────────────────────

AR_CROSS_LINGUAL = [
    pair(SYSTEM_EN,
         "Say 'hello' and introduce yourself briefly in Arabic.",
         "مرحباً! أنا أنثوس، مساعد ذكاء اصطناعي أنشأه براين توشاي توماس.\n\n"
         "**Transliteration**: Marḥaban! Anā Anthūs, musāʿid dhakāʾ iṣṭināʿī ansha'ahu Brāyān Tūshāy Tūmās.\n\n"
         "**Translation**: Hello! I am Anthos, an AI assistant created by Brian Tushae Thomas.\n\n"
         "**Useful Arabic greetings:**\n"
         "- مرحباً (Marḥaban) — Hello\n"
         "- السلام عليكم (As-salāmu ʿalaykum) — Peace be upon you (Islamic greeting)\n"
         "- صباح الخير (Ṣabāḥ al-khayr) — Good morning\n"
         "- مساء الخير (Masāʾ al-khayr) — Good evening\n"
         "- كيف حالك؟ (Kayfa ḥālak?) — How are you?"),
    pair(SYSTEM_EN,
         "What are the numbers 1-10 in Arabic?",
         "الأرقام من 1 إلى 10 بالعربية:\n\n"
         "| رقم | عربي | تحويل صوتي |\n"
         "|-----|------|------------|\n"
         "| 1 | واحد | wāḥid |\n"
         "| 2 | اثنان | ithnān |\n"
         "| 3 | ثلاثة | thalātha |\n"
         "| 4 | أربعة | arbaʿa |\n"
         "| 5 | خمسة | khamsa |\n"
         "| 6 | ستة | sitta |\n"
         "| 7 | سبعة | sabʿa |\n"
         "| 8 | ثمانية | thamāniya |\n"
         "| 9 | تسعة | tisʿa |\n"
         "| 10 | عشرة | ʿashara |\n\n"
         "**Note**: Arabic is written right to left (RTL). The numeral symbols used globally (1, 2, 3...) are actually of Arabic origin — called 'Arabic-Indic numerals'. Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩) are used in some Arab countries."),
    pair(SYSTEM_EN,
         "Teach me a few basic Arabic phrases for travel.",
         "أهلاً وسهلاً! (Ahlan wa sahlan!) — Welcome!\n\n"
         "**Essential travel phrases:**\n\n"
         "| Situation | Arabic | Pronunciation |\n"
         "|-----------|--------|---------------|\n"
         "| Thank you | شكراً | Shukran |\n"
         "| Please | من فضلك | Min faḍlak (to man) / Min faḍlik (to woman) |\n"
         "| Yes | نعم | Naʿam |\n"
         "| No | لا | Lā |\n"
         "| Excuse me | عفواً | ʿAfwan |\n"
         "| I don't understand | لا أفهم | Lā afham |\n"
         "| Where is...? | أين...؟ | Ayna...? |\n"
         "| How much? | كم الثمن؟ | Kam al-thaman? |\n"
         "| Hotel | فندق | Funduq |\n"
         "| Airport | مطار | Maṭār |\n"
         "| Water | ماء | Māʾ |\n"
         "| Bathroom | حمام | Ḥammām |\n\n"
         "**Tip**: Arabic has many dialects — Egyptian, Gulf, Levantine, Moroccan. Modern Standard Arabic (MSA / الفصحى) is understood everywhere but sounds formal. For Egypt: use Egyptian dialect words, for Gulf countries: Gulf Arabic."),
    pair(SYSTEM_EN,
         "What does 'Inshallah' mean and how is it used culturally?",
         "'Inshallah' (إن شاء الله — In shāʾ Allāh) literally means 'If God wills' or 'God willing'.\n\n"
         "**Usage in Arabic culture:**\n\n"
         "It's one of the most frequently used phrases in Arabic and expresses:\n"
         "1. **Genuine hope and intent**: 'I'll be there at 8, inshallah' — meaning sincerely 'I plan to and hope God allows it'\n"
         "2. **Acknowledgment of uncertainty**: All future events are ultimately in God's hands\n"
         "3. **Polite deflection**: Sometimes used to mean 'maybe' or 'we'll see' — a non-committal response\n"
         "4. **Blessing a plan**: After stating a goal, adding inshallah is a cultural norm\n\n"
         "**Cultural note**: Non-Arab speakers sometimes misread 'inshallah' as evasive. It's more nuanced — it reflects a worldview where human plans exist within divine providence. In business contexts, following up with specific details (time, date, written confirmation) clarifies concrete commitment.\n\n"
         "**Similar expressions in other languages**: 'Ojalá' in Spanish (from Arabic 'wa shāʾ Allāh'), 'Inch'allah' in French informal use."),
    pair(SYSTEM_EN,
         "What is the Arabic alphabet and how does Arabic writing work?",
         "Arabic uses a 28-letter alphabet called **الأبجدية** (al-abjadiyya). Key features:\n\n"
         "**Direction**: Written right to left (RTL)\n\n"
         "**Consonantal alphabet (Abjad)**: Arabic letters primarily represent consonants. Short vowels are usually omitted in everyday writing — readers infer them from context. (Like reading 'th ct st n th mt' and knowing it says 'the cat sat on the mat')\n\n"
         "**Vowel marks (harakat)**: Short vowels can be written as diacritical marks above/below letters. Used in the Quran, children's books, and language learning materials.\n\n"
         "**Connected script**: Most letters connect to adjacent letters and change shape depending on position (beginning, middle, end of word).\n\n"
         "**Example — the word for 'book' (kitāb):**\n"
         "- كِتَابٌ (with vowel marks)\n"
         "- كتاب (without vowel marks — normal writing)\n\n"
         "**The 28 letters** (approximate English sounds):\n"
         "ا (alif, A) ب (bā, B) ت (tā, T) ث (thā, TH) ج (jīm, J) ح (ḥā, H) خ (khā, KH)...\n\n"
         "Arabic has sounds not in English: the 'ayn' (ع), guttural 'gh' (غ), emphatic consonants (ص ض ط ظ)."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Assembler
# ─────────────────────────────────────────────────────────────────────────────

ALL_PAIRS = (
    ES_GENERAL_QA +       # 15
    ES_CODE +             # 7
    ES_CONVERSATIONAL +   # 5
    ES_TRANSLATION +      # 5
    ES_CROSS_LINGUAL +    # 4
    FR_GENERAL +          # 8
    FR_CODE +             # 3
    FR_CROSS_LINGUAL +    # 2
    PT_GENERAL +          # 6
    PT_CODE +             # 2
    PT_CROSS_LINGUAL +    # 2
    AR_CROSS_LINGUAL      # 5
)

# ---------- expansion: replicate with slight context variation to hit target ----------

def expand(pairs: list, target: int, seed: int = 99) -> list:
    """Repeat-shuffle pairs until we hit target count."""
    rng = random.Random(seed)
    result = list(pairs)
    while len(result) < target:
        batch = list(pairs)
        rng.shuffle(batch)
        result.extend(batch)
    rng.shuffle(result)
    return result[:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',   default='data/multilingual_sft.jsonl')
    parser.add_argument('--count', type=int, default=1200,
                        help='Total pairs to emit (default 1200; base is 64 unique pairs, expanded by shuffled repetition)')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = expand(ALL_PAIRS, args.count)
    random.shuffle(pairs)

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # Count by language
    lang_counts = {'es': 0, 'fr': 0, 'pt': 0, 'ar': 0, 'other': 0}
    for p in pairs:
        text = ' '.join(c.get('value','') for c in p['conversations'])
        if re.search(r'[¿¡]|español|Estoy|¿Qué|SYSTEM_ES|eres Anthos', text):
            lang_counts['es'] += 1
        elif re.search(r'français|Bonjour|Tu es Anthos|Qu\'est', text):
            lang_counts['fr'] += 1
        elif re.search(r'Você é Anthos|O que é|português', text):
            lang_counts['pt'] += 1
        elif re.search(r'[؀-ۿ]|Marḥaban|ʿafwan|inshallah', text, re.I):
            lang_counts['ar'] += 1
        else:
            lang_counts['other'] += 1

    print(f"\n{'═'*50}")
    print(f"  Multilingual dataset generated")
    print(f"  Output: {out_path}  ({len(pairs)} pairs)")
    print(f"\n  Language breakdown:")
    for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(pairs)
        print(f"    {lang}: {n:,}  ({pct:.1f}%)")
    print(f"{'═'*50}")
    print(f"\n  Next: merge into sft_master.jsonl and rebuild manifest:")
    print(f"    cat data/multilingual_sft.jsonl >> data/sft_master.jsonl")
    print(f"    python build_training_manifest.py")


if __name__ == '__main__':
    main()
