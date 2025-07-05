# GB1
for pdb_path in "./example/GB1/2gi9.pdb";
do 
    mkdir -p $pdb_path-ligandmpnn ;
    python run.py \
            --seed 111 \
            --pdb_path $pdb_path \
            --out_folder $pdb_path-ligandmpnn \
            --redesigned_residues "A39 A40 A41 A54" \
            --batch_size 10 \
            --number_of_batches 5

done;