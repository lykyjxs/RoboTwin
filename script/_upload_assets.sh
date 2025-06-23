# Create upload_cache
cd assets
mkdir upload_cache

# Embodiments
mkdir upload_cache/embodiments_zip

cd embodiments

# for SUBFOLDER in "."/*; do
#     if [ -d "$SUBFOLDER" ]; then  
#         FOLDER_NAME=$(basename "$SUBFOLDER") 
#         ZIP_FILE="../upload_cache/embodiments_zip/${FOLDER_NAME}.zip" 

#         zip -r "$ZIP_FILE" "$SUBFOLDER"
#         echo "Compressed $SUBFOLDER to $ZIP_FILE"
#     fi
# done

cd ..

# Objects
# zip -r upload_cache/objects.zip objects 

# # Messy Objects
# zip -r upload_cache/messy_objects.zip messy_objects

# # Textures
# zip -r upload_cache/textures.zip textures

# Upload Files
python _upload.py

# Remove upload_cache
cd ..
rm -rf upload_cache