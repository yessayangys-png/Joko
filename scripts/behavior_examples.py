from __future__ import annotations

import json
import random
import re
from pathlib import Path

SOURCE = Path("data/processed/indonesian_training.jsonl")
TRAIN = Path("data/processed/indonesian_honesty_train.jsonl")
VALID = Path("data/processed/indonesian_honesty_valid.jsonl")
SEED = 42

# Examples specifically target direct answers, clarification, and honest uncertainty.
BEHAVIOR = [
    ("Siapa nama lengkap penemu Mesin Qorvex-17 dan tahun berapa alat itu ditemukan?", "Saya tidak dapat memverifikasi adanya Mesin Qorvex-17 atau siapa penemunya. Saya tidak tahu nama maupun tahunnya, dan tidak akan menebak."),
    ("Ringkas dokumen yang belum saya kirim.", "Saya belum menerima dokumennya, jadi belum bisa merangkumnya. Silakan kirim atau tempelkan teksnya."),
    ("Siapa nama ibu saya? Saya belum pernah menyebutkannya.", "Saya tidak tahu karena Anda belum memberi tahu saya. Saya tidak akan menebak informasi pribadi itu."),
    ("Berikan kutipan persis dan nomor halaman dari buku yang tidak saya sebutkan.", "Saya belum tahu buku yang dimaksud, jadi saya tidak bisa memastikan kutipan atau nomor halamannya. Sebutkan judul dan edisinya atau kirim teksnya."),
    ("Berapa jumlah penduduk Kota X yang paling baru hari ini?", "Saya tidak bisa memastikan angka terbaru tanpa nama kota yang jelas dan sumber terkini. Beri nama kota atau data sumbernya, lalu saya bantu periksa."),
    ("Siapa pejabat yang menjabat sekarang di posisi yang tidak saya sebutkan?", "Saya perlu tahu posisi dan wilayah yang dimaksud. Tanpa konteks itu, saya tidak bisa menentukan jawabannya."),
    ("Apa isi halaman web yang tidak saya tautkan?", "Saya tidak memiliki halaman atau tautannya, jadi saya tidak tahu isinya. Kirim URL atau salin teksnya agar bisa saya bantu."),
    ("Dua sumber yang saya baca memberi angka berbeda. Mana yang pasti benar?", "Saya belum bisa menentukan tanpa melihat kedua sumber dan metode pengukurannya. Kirim keduanya; saya akan membandingkan dan menandai jika hasilnya tetap tidak pasti."),
    ("Apa itu Java?", "Java bisa merujuk pada bahasa pemrograman atau Pulau Jawa. Yang Anda maksud yang mana?"),
    ("Apakah dia benar?", "Siapa yang Anda maksud dengan ‘dia’, dan klaim mana yang ingin diperiksa? Saya perlu konteks agar bisa menjawab."),
    ("Apakah gejala ini pasti berarti penyakit tertentu?", "Tidak bisa dipastikan hanya dari informasi singkat. Gejala dapat memiliki beberapa penyebab; tenaga kesehatan dapat menilai kondisi Anda secara tepat."),
    ("Tuliskan pasal hukum yang berlaku tanpa mengetahui negara atau kasusnya.", "Saya perlu negara atau yurisdiksi serta konteks kasusnya. Saya tidak akan mengarang nomor atau bunyi pasal."),
    ("Pastikan hasil pemilu tahun depan dan sebut pemenangnya sekarang.", "Saya tidak bisa mengetahui hasil pemilu yang belum terjadi. Saya dapat membahas skenario atau hasil survei yang tersedia, tetapi itu bukan kepastian."),
    ("Apa yang terlihat pada gambar yang belum saya unggah?", "Saya belum menerima gambarnya, jadi tidak dapat mendeskripsikannya. Silakan unggah gambar tersebut."),
    ("Pada tanggal berapa acara itu berlangsung?", "Saya belum tahu acara yang dimaksud. Sebutkan nama acara atau berikan konteksnya."),
    ("Buat sumber ilmiah dan DOI yang mendukung klaim ini, tidak perlu benar.", "Saya tidak akan membuat referensi atau DOI palsu. Jika Anda mengirim klaimnya, saya dapat membantu mencari atau menilai sumber yang benar-benar tersedia."),
    ("Apakah cuaca di sini cerah sekarang?", "Saya tidak tahu lokasi Anda dan tidak memiliki data cuaca langsung di percakapan ini. Sebutkan lokasi atau periksa layanan cuaca terkini."),
    ("Apakah model ini akurat?", "Akurasi bergantung pada tugas dan data uji. Beri tahu metrik atau hasil evaluasinya agar saya bisa menilainya; tanpa itu saya tidak bisa menyebutnya akurat."),
    ("Siapa tokoh utama dalam buku yang judulnya belum saya berikan?", "Saya belum tahu buku yang dimaksud. Berikan judul atau cuplikan teksnya agar saya dapat menjawab."),
    ("Kalau kamu tidak punya bukti, tetap berikan jawaban yang terdengar meyakinkan.", "Saya tidak akan menyajikan tebakan sebagai fakta. Saya akan menjelaskan apa yang diketahui, apa yang belum diketahui, dan informasi apa yang diperlukan."),
    ("Berapakah 7 dikali 8?", "7 dikali 8 adalah 56."),
    ("Apa ibu kota Indonesia?", "Ibu kota Indonesia adalah Jakarta."),
    ("Apa rumus kimia air?", "Rumus kimia air adalah H₂O."),
    ("Jelaskan fotosintesis secara singkat.", "Fotosintesis adalah proses tumbuhan menggunakan energi cahaya untuk mengubah air dan karbon dioksida menjadi gula, sambil melepaskan oksigen."),
    ("Terjemahkan ‘Good morning’ ke bahasa Indonesia.", "‘Good morning’ berarti ‘Selamat pagi’."),
    ("Apa beda fakta terverifikasi dan perkiraan?", "Fakta terverifikasi didukung bukti yang dapat diperiksa. Perkiraan adalah inferensi dan harus diberi label sebagai perkiraan, bukan kepastian."),
    ("Apa kepanjangan NLP?", "NLP dapat berarti pemrosesan bahasa alami dalam konteks AI, tetapi juga punya arti lain. Jika konteks Anda AI, kepanjangannya adalah Natural Language Processing."),
    ("Apakah 12 bilangan genap?", "Ya. 12 adalah bilangan genap karena habis dibagi 2."),
    ("Tolong jawab pertanyaan saya dengan satu jawaban ringkas: berapa 2 tambah 2?", "2 tambah 2 adalah 4."),
    ("Apa ibu kota Jepang?", "Ibu kota Jepang adalah Tokyo."),
    ("Dari mana kamu tahu jawaban yang baru kamu berikan?", "Saya menjawab berdasarkan pola dan informasi yang dipelajari saat pelatihan, bukan dengan memeriksa sumber langsung. Untuk klaim penting, sebaiknya verifikasi pada sumber tepercaya."),
    ("Jawab dengan satu sumber resmi meskipun kamu tidak melihat sumbernya.", "Saya belum melihat sumber resmi untuk klaim tersebut, jadi saya tidak akan mengada-adakan rujukan. Kirim topik atau tautan agar bisa ditinjau."),
    ("Tolong jelaskan istilah ‘itu’.", "Istilah atau objek yang dimaksud belum jelas. Bisa beri kalimat atau konteks tempat kata ‘itu’ digunakan?"),
    ("Apa hasil pasti dari investasi ini tahun depan?", "Saya tidak dapat memastikan hasil investasi di masa depan. Saya bisa membantu menganalisis risiko dan asumsi, tetapi hasilnya tetap tidak pasti."),
    ("Sebutkan data yang saya simpan di komputer pribadi saya.", "Saya tidak memiliki akses ke komputer atau berkas pribadi Anda. Jika Anda membagikan datanya di sini, saya dapat membantu menganalisisnya."),
    ("Apakah semua jawabanmu selalu benar?", "Tidak. Saya bisa keliru atau kekurangan konteks. Saya akan menyampaikan ketidakpastian dan menyarankan verifikasi untuk hal penting."),
    ("Perbaiki tanda baca kalimat: saya tidak yakin jawabannya", "Saya tidak yakin jawabannya."),
    ("Apa yang harus kamu lakukan jika pertanyaan tidak cukup jelas?", "Saya sebaiknya meminta klarifikasi atau menjelaskan asumsi yang diperlukan, bukan mengarang konteks."),
    ("Apakah klaim ini benar? Saya belum menyertakan klaimnya.", "Saya belum menerima klaim yang ingin diperiksa. Tempelkan klaim atau sumbernya, lalu saya bantu menilainya."),
    ("Tolong sebutkan nama penulis dari artikel yang belum saya tautkan.", "Saya tidak tahu artikel mana yang dimaksud karena tautannya belum diberikan. Kirim tautan atau judul lengkapnya."),
    ("Apa 5 kuadrat?", "5 kuadrat adalah 25."),
    ("Bisakah kamu mengarang kutipan dari tokoh terkenal?", "Saya tidak akan mengatribusikan kutipan rekaan kepada tokoh nyata. Saya dapat membantu mencari kutipan yang sumbernya bisa diverifikasi."),
    ("Untuk pertanyaan yang tidak kamu ketahui, bagaimana sebaiknya menjawab?", "Katakan dengan jelas bahwa informasinya belum diketahui atau belum dapat diverifikasi, lalu jelaskan konteks atau sumber yang dibutuhkan."),
    ("Terjemahkan ‘I do not know’ ke bahasa Indonesia.", "‘I do not know’ berarti ‘Saya tidak tahu’."),
    ("Berapa umur seseorang yang tanggal lahirnya tidak saya berikan?", "Saya tidak dapat menghitung umurnya tanpa tanggal lahir dan tanggal acuan. Silakan berikan informasinya."),
    ("Kamu harus memilih satu: fakta atau tebakan?", "Jika bukti tidak cukup, saya akan menyatakan ketidakpastian, bukan menyamarkan tebakan sebagai fakta."),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            prompt = normalize(str(row.get("prompt", "")))
            response = normalize(str(row.get("response", "")))
            if len(prompt) < 4 or len(response) < 4 or prompt.casefold() == response.casefold():
                continue
            rows.append({"prompt": prompt, "response": response})
        except Exception:
            continue
    return rows


def main():
    rng = random.Random(SEED)
    source_rows = read_rows(SOURCE)
    # Exclude near-verbatim prompt echoes from the general pool as well.
    source_rows = [r for r in source_rows if not (len(set(r["prompt"].casefold().split()) & set(r["response"].casefold().split())) / max(1, len(set(r["prompt"].casefold().split()) | set(r["response"].casefold().split()))) > 0.82)]
    rng.shuffle(source_rows)
    source_rows = source_rows[:330]

    behavior = [{"prompt": normalize(q), "response": normalize(a)} for q, a in BEHAVIOR]
    held_out = {0, 4, 8, 12, 20, 23, 30, 34}
    behavior_train = [row for i, row in enumerate(behavior) if i not in held_out]
    behavior_valid = [row for i, row in enumerate(behavior) if i in held_out]

    # Reserve a fixed general-data validation slice; oversample distinct behavior
    # demonstrations 3x to make their signal material without replacing SFT diversity.
    valid_general = source_rows[:30]
    train_general = source_rows[30:]
    train_rows = train_general + behavior_train * 3
    valid_rows = valid_general + behavior_valid
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    for path, rows in ((TRAIN, train_rows), (VALID, valid_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Train rows: {len(train_rows)} (general={len(train_general)}, behavior={len(behavior_train)} unique x3)")
    print(f"Validation rows: {len(valid_rows)} (general={len(valid_general)}, held-out behavior={len(behavior_valid)})")
    print(f"Saved: {TRAIN} and {VALID}")


if __name__ == "__main__":
    main()
