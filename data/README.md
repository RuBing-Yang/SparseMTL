img2dataset --url_list='download_urls.txt' --output_folder='./suite-sparse/origin'  --thread_count=64 --incremental_mode=incremental --resize_mode=no --disable_all_reencoding

img2dataset --url_list='test_urls.txt' --output_folder='./suite-sparse/test'  --thread_count=64 --incremental_mode=incremental --resize_mode=no --disable_all_reencoding
