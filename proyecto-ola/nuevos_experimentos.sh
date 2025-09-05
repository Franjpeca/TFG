#!/bin/bash

echo "======== NNOP - SEED 42 ========"
kedro run --tags model_NNOP --params "training_settings.seed=42" | tee -a /mnt/d/nuevos_lanzamientos/nnop/training_seed_42.log
echo "======== NNOP - SEED 43 ========"
kedro run --tags model_NNOP --params "training_settings.seed=43" | tee -a /mnt/d/nuevos_lanzamientos/nnop/training_seed_43.log
echo "======== NNOP - SEED 44 ========"
kedro run --tags model_NNOP --params "training_settings.seed=44" | tee -a /mnt/d/nuevos_lanzamientos/nnop/training_seed_44.log
echo "======== NNOP - SEED 45 ========"
kedro run --tags model_NNOP --params "training_settings.seed=45" | tee -a /mnt/d/nuevos_lanzamientos/nnop/training_seed_45.log
echo "======== NNOP - SEED 46 ========"
kedro run --tags model_NNOP --params "training_settings.seed=46" | tee -a /mnt/d/nuevos_lanzamientos/nnop/training_seed_46.log
echo "======== NNPOM - SEED 42 ========"
kedro run --tags model_NNPOM --params "training_settings.seed=42" | tee -a /mnt/d/nuevos_lanzamientos/nnpom/training_seed_42.log
echo "======== NNPOM - SEED 43 ========"
kedro run --tags model_NNPOM --params "training_settings.seed=43" | tee -a /mnt/d/nuevos_lanzamientos/nnpom/training_seed_43.log
echo "======== NNPOM - SEED 44 ========"
kedro run --tags model_NNPOM --params "training_settings.seed=44" | tee -a /mnt/d/nuevos_lanzamientos/nnpom/training_seed_44.log
echo "======== NNPOM - SEED 45 ========"
kedro run --tags model_NNPOM --params "training_settings.seed=45" | tee -a /mnt/d/nuevos_lanzamientos/nnpom/training_seed_45.log
echo "======== NNPOM - SEED 46 ========"
kedro run --tags model_NNPOM --params "training_settings.seed=46" | tee -a /mnt/d/nuevos_lanzamientos/nnpom/training_seed_46.log
