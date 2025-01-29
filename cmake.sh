python -m venv conway_env
source conway_env/bin/activate
pip install -r requirements.txt

cmake -S . -B build -G Ninja # configures the project for building
cmake --build build # actually builds the project
# /!\ don't forget to re-build whenever you modify the cpp/cuda code /!\