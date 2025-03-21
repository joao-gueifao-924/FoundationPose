DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

# Disable Kaolin and BundleSDF as we are not doing model-free object pose estimation.
# We currently only perform object pose estimation given their 3D models are provided as input. 
#cd /kaolin && rm -rf build *egg* && pip install -e .
#cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
