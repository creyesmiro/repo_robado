# IMA543: Redes Neuronales de Aprendizaje Profundo

* **Alejandro Vega Alarc�n**
* **Departamento de Ingenier�a Matem�ticas y Minor de An�lisis de Datos**
* **Segundo semestre 2023**

## Revisi�n Git
  - Git s un sistema de control de versiones gratis y de c�digo libre. F�cil de aprender
  a usar y conveniente para el manejo de proyectos.

  - https://git-scm.com/

  ### Crear una cuenta en github
  - https://github.com/
  ### Crear un token personal
  ir a *Settings* > *Developer settings* > *Personal access tokens* > *Classic*
  - no olvidar guardar el token en un .txt para usarlo despues
  - en caso de que se pierda se pude resetear en las configuraciones 
  ### Guardar las credenciales localmente  
  - para configurar el nombre de usuario:
    
  ```git config --global username "user_name" ```
  
  - para configurar el email:
    
  ``` git config --global user.email "eser_email"```
  
  - la contrase�a(token) se ajusta mas tarde
  ### Para crear un repositorio local
  - crear un nuevo repo en github y guardar el <link>
  - para inicializar git en un directorio:
    
  ```git init```
      
  - para a�adir archivos     
    
  ```git add .```
     
  ```git add ejemplo1.py ejemplo2.txt ejemplo3.md```
  
  - se hace un commit con un comentario para empaquetar antes de subir  

  ```git commit -m "Initial commit"```
      
  - para enlazar el directorio local al repositorio en github
    
  ```git remote add origin <link>```
  
  - para subir los datos al repo en la rama "master"
    
  ```git push -u origin master```
      
  ### para volver a una version anterior  

  * revisar los comits
    ```git log```

    revisamos el *hash* del commit al que nos interesa volver

  * borrar todos los cambios a partir de x commit:
    ```git reset --hard commit-hash```

  * deshacer a partir a partir de un commit x dejando los cambios que habian como un
    nuevo commit para hacerle modificaciones:
    ```git reset --soft commit-hash```
  
  * deshacer los cambios a partir de un commit x pero dejando los cambios en el
    directorio de trabajo:
    ```git reset```
  
  - como a�adir colaboradores a un repo privado

## Conexi�n a Khipu
* direcci�n del servidor: 200.13.6.14
* puerto: 10022
  ### Para conectarse al servidor:
  - En la terminal:
      ```ssh nombre_usuario@200.13.6.14 -p 10022```
  - Luego la clave del usuario
  ### Bajar un erpositorio en el servidor:
    ```git clone url_repo```
  - Ojo si el repositorio es p�blico o privado, si es privado
    necesitan permiso(creo) e ingresar sus credenciales. 
  - Khipu tiene git instalado. 
  ### Manejo de screen
  - Screen es una herramienta para manejar seciones en linux, �til 
    por si la conexi�n es inestable o se requiere ejecutar multiples comandos.
  - Para abrir screen en el servidor:
      ```screen```
  - Una vez abierto el programa pueden presionar los comandos:
      * ctrl+a c //para crear una nueva "pesa�a"
      + ctrl+a n,p //para navegar entre las pesta��as disonibles
  ### Checkeo de recursos
  - Para revisar el uso de las CPUs del servidor
      ```htop```
  - Para revisar el uso de las GPUs del servidor
      ```watch -n 1 nvidia-smi```
      este comando actualiza la informacion cada 1 seg
  ### Python
  - Python no est� activado por default en la consola de Khipu
      ```module load conda/3-py39_4.12.0```
      para cargar python.
  - Para crear el entorno del ramo hay que moverse a la carpeta 
      que contiene el archivo con la informacion de las librerias.
      ```cd nombre_repo```
      luego ejecutar
      ```conda env create -f ima543Env.yml```
  - Teniendo ya el entorno listo, podemos ejecutar nuestra rutina
      ```python ejemplo.py```

## Uso de m�ltiples GPUs con Keras
El entrenamiento con m�ltiples GPUs nos consiste en dividir el batch
de datos de entrenamiento para que las GPUs realizen los calculos en 
pralelo y obtengan un gradiente de manera democr�tica para ajustar
el modelo.
  ### C�mo usar
  - Se define una instancia de *tf.distribute.MirrorredStrategy()*
    en donde se pasan las GPUs en las que se quieren distribuir los
    datos como argumento.
  - Las GPUs se pasan como argumento en una lista:  
      ```["/gpu:{0}", ..., "/gpu:{n}"]```
  - Dentro del bloque definido por strategy.scope() deben ir
      el modelo, la perdida, el compilador, las variables que 
      se quieren distribuir.
  - Pueden quedar fuera: model.fit(), model.evaluate(), la creaci�n
      de los input dataset, la definici�n de los pasos de entrenamiento,
      guardado y checkpoints del modelo.
  ### Se podr�a usar para paralelizar la evaluaci�n del modelo?
  ### Documentaci�n:
  * MirroredStrategy : https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy
  * Strategy#scope : https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#scope
  * distribute/keras : https://www.tensorflow.org/tutorials/distribute/keras
  * RNN : https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN 
  * GRU : https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU




  
  