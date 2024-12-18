from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import base64
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Muat model
model = tf.keras.models.load_model('final_garbage_classification_model.keras')

# Label klasifikasi sampah
labels = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "residu", "white-glass"
]

# Informasi dan kategori sampah (persis seperti di Streamlit)
label_information = {
    "battery": {
        "deskripsi": "Baterai bekas termasuk sampah elektronik yang mengandung bahan kimia berbahaya seperti timbal dan merkuri.",
        "penanganan": "Baterai bekas harus dikumpulkan dan didaur ulang melalui pusat daur ulang elektronik.",
        "kategori": "B3 (Bahan Berbahaya dan Beracun)"
    },
    "biological": {
        "deskripsi": "Sampah biologis berasal dari sisa makhluk hidup seperti sisa makanan dan daun-daunan.",
        "penanganan": "Sampah ini dapat diolah menjadi kompos untuk pupuk alami.",
        "kategori": "Organik"
    },
    "brown-glass": {
        "deskripsi": "Sampah kaca berwarna coklat seperti botol minuman bekas.",
        "penanganan": "Pisahkan kaca berwarna dari jenis kaca lain dan kirimkan ke pusat daur ulang kaca.",
        "kategori": "Anorganik"
    },
    "cardboard": {
        "deskripsi": "Kardus atau kertas tebal bekas yang umum digunakan sebagai kemasan.",
        "penanganan": "Lipat dan kumpulkan kardus untuk didaur ulang menjadi produk kertas baru.",
        "kategori": "Anorganik"
    },
    "clothes": {
        "deskripsi": "Pakaian bekas yang sudah tidak digunakan.",
        "penanganan": "Sumbangkan pakaian layak pakai atau gunakan kembali sebagai kain lap.",
        "kategori": "Anorganik"
    },
    "green-glass": {
        "deskripsi": "Sampah kaca berwarna hijau seperti botol minuman.",
        "penanganan": "Pisahkan dan daur ulang bersama kaca berwarna lainnya.",
        "kategori": "Anorganik"
    },
    "metal": {
        "deskripsi": "Logam seperti kaleng minuman, besi tua, atau aluminium.",
        "penanganan": "Logam dapat dilebur kembali dan digunakan untuk pembuatan produk baru.",
        "kategori": "Anorganik"
    },
    "paper": {
        "deskripsi": "Sampah kertas seperti koran, majalah, atau kertas bekas.",
        "penanganan": "Kumpulkan dan daur ulang menjadi kertas daur ulang.",
        "kategori": "Anorganik"
    },
    "plastic": {
        "deskripsi": "Sampah plastik termasuk botol, kantong plastik, dan sedotan.",
        "penanganan": "Pisahkan plastik berdasarkan jenisnya dan kirim ke fasilitas daur ulang.",
        "kategori": "Anorganik"
    },
    "shoes": {
        "deskripsi": "Sepatu bekas yang sudah tidak layak digunakan.",
        "penanganan": "Sepatu bekas dapat disumbangkan atau didaur ulang menjadi bahan lain.",
        "kategori": "Anorganik"
    },
    "residu": {
        "deskripsi": "Sampah umum yang tidak dapat didaur ulang atau digunakan kembali.",
        "penanganan": "Buang ke tempat sampah akhir atau gunakan pengelolaan sampah terorganisir.",
        "kategori": "Residu"
    },
    "white-glass": {
        "deskripsi": "Sampah kaca bening seperti botol kaca putih atau gelas.",
        "penanganan": "Pisahkan kaca bening dan kirim ke pusat daur ulang kaca.",
        "kategori": "Anorganik"
    }
}

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Terima base64 image
        image_data = request.json['image']
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocessing
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prediksi
        predictions = model.predict(image_array)
        
        # Ambil label dengan probabilitas tertinggi
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Ambil label
        label = labels[label_index]
        
        # Ambil informasi tambahan dari label_information
        info = label_information.get(label, {
            "deskripsi": "Informasi tidak tersedia",
            "penanganan": "Informasi tidak tersedia",
            "kategori": "Tidak diketahui"
        })
        
        return jsonify({
            'label': label,
            'confidence': float(confidence),
            'deskripsi': info['deskripsi'],
            'penanganan': info['penanganan'],
            'kategori': info['kategori']
        })
    
    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
