## 53 exo
# region 1 A 218-226
# region 2 A 291-292
# region 3 A 502-515
# region1="A218 A219 A220 A221 A222 A223 A224 A225 A226"
# region2="A291 A292"
# region3="A502 A503 A504 A506 A507 A508 A509 A511 A512 A513 A515"

# 5-3 exonuclease
for pdb_path in "./example/taqP_fix_residues/1bgx_taq_53_exo_processed.pdb" "./example/taqP_fix_residues/1taq_53_exo_processed.pdb";
do 
    mkdir -p $pdb_path-region1 ;
    python run.py \
            --seed 111 \
            --pdb_path $pdb_path \
            --out_folder $pdb_path-region1 \
            --redesigned_residues "A218 A219 A220 A221 A222 A223 A224 A225 A226" \
            --batch_size 10 \
            --number_of_batches 5

    mkdir -p $pdb_path-region2 ;
    python run.py \
            --seed 111 \
            --pdb_path $pdb_path \
            --out_folder $pdb_path-region2 \
            --redesigned_residues "A291 A292" \
            --batch_size 10 \
            --number_of_batches 5
done;


## Polymerase
for pdb_path in "./example/taqP_fix_residues/pdb1ktq_processed.pdb" "./example/taqP_fix_residues/pdb4ktq_processed.pdb";
do  
    mkdir -p $pdb_path-region3 ;
    python run.py \
            --seed 111 \
            --pdb_path $pdb_path \
            --out_folder $pdb_path-region3 \
            --redesigned_residues "A502 A503 A504 A506 A507 A508 A509 A511 A512 A513 A515" \
            --batch_size 10 \
            --number_of_batches 5
done;
