#!/usr/bin/env bash
# El script compila las extensiones C/C++ de ORCA-Python (svorex y libsvmRank)

# Ir a la carpeta donde esta el script
cd "$(dirname "$0")"

# Comprobar si Python esta disponible
if ! command -v python > /dev/null; then
  echo "Error: No se encontro 'python'. Activa el entorno virtual o instala Python."
  exit 1
fi

# Mostrar version y ruta de Python
python -c "import sys; print('Python:', sys.version.split()[0], '| Prefix:', sys.prefix)"

# Aviso por si no esta activado el entorno
echo "Nota: Asegurate de tener el entorno virtual activado."

# Instalar compiladores y cabeceras (segun el gestor de paquetes)
if command -v apt > /dev/null; then
  echo "Instalando build-essential y python3-dev con apt..."
  sudo apt update
  sudo apt install -y build-essential python3-dev
elif command -v dnf > /dev/null; then
  echo "Instalando gcc y python3-devel con dnf..."
  sudo dnf install -y gcc gcc-c++ make python3-devel
elif command -v pacman > /dev/null; then
  echo "Instalando base-devel y python con pacman..."
  sudo pacman -Sy --noconfirm base-devel python
elif command -v zypper > /dev/null; then
  echo "Instalando gcc y python3-devel con zypper..."
  sudo zypper install -y gcc gcc-c++ make python3-devel
else
  echo "No se detecto un gestor de paquetes conocido. Instala un compilador C/C++ y cabeceras de Python manualmente si falla."
fi

# Instalar herramientas de Python necesarias
python -m pip install --upgrade pip setuptools wheel
export SETUPTOOLS_USE_DISTUTILS=local

# Compilar svorex
cd orca-python/orca_python/classifiers/svorex
python setup.py build_ext --inplace
cd -

# Compilar libsvmRank
cd orca-python/orca_python/classifiers/libsvmRank/python
python setup.py build_ext --inplace
cd -

# Mostrar ficheros .so generados
echo "Ficheros compilados (.so):"
find orca-python -maxdepth 3 -name "*.so"
echo "Listo."