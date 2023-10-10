import ctranslate2
import transformers
import numpy as np
import torch

model_path = "D:\\kompas-dev\\ai\\models\\indobert-base-uncased-ct2"
model_name = "indolem/indobert-base-uncased"
device = "cpu"

def embedder_v2(article_text: str):
    # Initialize the encoder and tokenizer
    encoder = ctranslate2.Encoder(model_path, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Tokenize the article into sentences
    sentences = article_text.split('.')

    # Initialize a list to store embeddings
    embeddings = []

    # Initialize a running count of tokens
    running_token_count = 0

    # Process each sentence and generate embeddings
    for sentence in sentences:
        text = [sentence]
        # Tokenize the sentence
        tokens = tokenizer(text).input_ids

        # Check if the tokens exceed the 512-token limit
        if running_token_count + len(tokens) <= 512:
            # Update the running token count
            running_token_count += len(tokens)

            # Encode the tokens and obtain the pooled output
            output = encoder.forward_batch(tokens)
            pooler_output = np.array(output.pooler_output)
            pooler_output = torch.as_tensor(pooler_output)

            # Append the pooled output to the embeddings list
            formatted_vector = [float(val.item()) for val in pooler_output[0]]

            embeddings.append(formatted_vector)

        else:
            break  # Stop processing sentences once the 512-token limit is reached
    
    concatenated_embeddings = np.concatenate(embeddings, axis=0)

    return concatenated_embeddings

def embedder(text: str):

    encoder = ctranslate2.Encoder(model_path, device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    inputs = [text]

    tokens = tokenizer(inputs, truncation=True, max_length=512, padding=True).input_ids

    output = encoder.forward_batch(tokens)
    pooler_output = output.pooler_output

    pooler_output = np.array(pooler_output)
    pooler_output = torch.as_tensor(pooler_output)

    formatted_vector = [float(val.item()) for val in pooler_output[0]]

    return formatted_vector

article_text = "4 Jurusan Akuntansi Terbaik di Indonesia, Sekian Biaya Kuliahnya. Berikut 4 jurusan Akuntansi terbaik di Indonesia versi QS World University by Subject 2023. Intip juga biaya Jika bidang Saintek memiliki Jurusan Kedokteran sebagai jurusan dengan peminat terbanyak, maka bidang sosial humaniora (soshum) memiliki Jurusan Akuntansi yang menjadi salah satu jurusan terfavorit dalam seleksi penerimaan mahasiswa baru setiap tahunnya. Bagi kamu yang menyukai hitung-hitungan, jurusan ini bisa menjadi salah satu pilihanmu ketika berkuliah nanti. Di Indonesia sendiri, terdapat banyak kampus yang membuka jurusan Akuntansi. Di Indonesia, ada 4 jurusan Akuntansi yang masuk pemeringkatan QS World University by Subject 2023. Peringkat ini mencakup total 54 disiplin ilmu, yang dikelompokkan ke dalam lima bidang studi, yaitu Seni dan Humaniora, Rekayasa dan Teknologi, Ilmu Hayati dan Kedokteran, Ilmu Pengetahuan Alam, serta Ilmu dan Manajemen Sosial. Berikut 4 jurusan akuntasi yang masuk pemeringkatan QS World University by Subject 2023 dan kisaran biaya kuliahnya: 1. Universitas Airlangga (Unair) Peringkat dunia: 101-150 Uang Kuliah Tunggal (UKT) Akuntansi 2023 jalur SNBP dan SNBT 1A: Rp 0 - Rp 500.000 1B: Rp 1 juta 1C: Rp 2,4 juta 2: Rp 6 juta 3: Rp 7,5 juta 4: Rp 10 juta Jalur mandiri reguler Uang Kuliah Semester (UKS): Rp 7,5 juta/semester Uang Kuliah Awal (UKA): minimal Rp 45 juta, dibayar 1 kali saat masuk Jalur mandiri kemitraan UKS: Rp 7,5 juta UKA: minimal Rp 75 juta 2. Universitas Indonesia (UI) Peringkat dunia: 101-150 Kelas reguler Biaya pendidikan (Uang Kuliah Tunggal) yang dibayarkan setiap semester dapat disesuaikan dengan kemampuan penanggung biaya pendidikan tanpa harus membayar Iuran Pembangunan Institusi (IPI). Uang Kuliah Tunggal (UKT) terdiri atas 11 kelas yang disusun berdasarkan prinsip berkeadilan. Kelas 1: Rp 0 - Rp 500.000 Kelas 2: Rp 500.000 - Rp 1 juta Kelas 3: Rp 1 juta - Rp 2 juta Kelas 4: Rp 2 juta - Rp 3 juta Kelas 5: Rp 3 juta - Rp 4 juta Kelas 6: Rp 4 juta - Rp 5 juta Kelas 7: Rp 5 juta - Rp 7,5 juta Kelas 8: Rp 7,5 juta - Rp 10 juta Kelas 9: Rp 10 juta - Rp 12,5 juta Kelas 10: Rp 12,5 juta- Rp 15 juta Kelas 11: Rp 15 juta - Rp 17,5 juta Program nonreguler Biaya pendidikan (Uang Kuliah Tunggal) meliputi Iuran Pembangunan Institusi (IPI) yang dibayarkan satu kali dan uang kuliah yang dibayarkan per semester bersifat tetap. Biaya Pendidikan: Rp15 juta Iuran Pembangunan Institusi (IPI): Rp 34 juta Kelas internasional Biaya pendidikan (Uang Kuliah Tunggal) meliputi Iuran Pembangunan Institusi (IPI) yang dibayarkan satu kali dan uang kuliah yang dibayarkan per semester bersifat tetap. Biaya pendidikan: Rp 36 juta IPI: Rp 38 juta 3. Universitas Gadjah Mada (UGM) Pada tahun 2023, berikut UKT jalur SNBP, SNBT dan mandiri Kedokteran di UGM: UKT Pendidikan Unggul: Rp 9,2 juta UKT Pendidikan Unggul Bersubsidi 25 persen: Rp 6,9 juta UKT Pendidikan Unggul Bersubsidi 50 persen: Rp 4,6 juta UKT Pendidikan Unggul Bersubsidi 75 persen: Rp 2,3 juta UKT Pendidikan Unggul Bersubsidi 100 persen: Rp 0 Bagi calon mahasiswa yang diterima melalui jalur UM-CBT UGM pada tahun akademik 2023/2024 dan ditetapkan UKT Pendidikan Unggul (memiliki kemampuan ekonomi baik) dikenakan Sumbangan Solidaritas Pendidikan Unggul (SSPU) di UGM sebesar Rp 20 juta untuk bidang Ilmu Sosial dan Humaniora. 4. Universitas Sebelas Maret (UNS) Peringkat dunia: 301-330 UKT Akuntansi di UNS UKT 1: Rp 475.500 UKT 2: Rp 975.500 UKT 3: Rp 3,225,5 juta UKT 4: Rp 5,225,5 juta UKT 5: Rp 5,715,5 juta UKT 6: 6,205,5 juta UKT 7: Rp 6,705,5 juta UKT 8: Rp 7,475,5 juta Sumbangan Pengembangan Institusi (SPI) jalur mandiri Kelompok 1: Rp 10 juta Kelompok 2: Rp 25 juta Kelompok 3: Rp 40 juta Kelompok 4: > Rp 40 juta"

if __name__ == "__main__":
    embedder(article_text)