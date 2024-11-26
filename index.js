const tf = require('@tensorflow/tfjs');

// Dataset (Data Mentah)
const data = [
  { usia: 25, jenisKelamin: 1, pendidikan: 4, pengeluaran: 5, keluarga: 3, lokasi: 1, pendapatan: 120 },
  { usia: 40, jenisKelamin: 0, pendidikan: 5, pengeluaran: 10, keluarga: 4, lokasi: 1, pendapatan: 300 },
  { usia: 35, jenisKelamin: 1, pendidikan: 3, pengeluaran: 6, keluarga: 5, lokasi: 2, pendapatan: 200 },
  { usia: 29, jenisKelamin: 0, pendidikan: 2, pengeluaran: 3, keluarga: 2, lokasi: 3, pendapatan: 80 },
  { usia: 50, jenisKelamin: 1, pendidikan: 6, pengeluaran: 20, keluarga: 2, lokasi: 1, pendapatan: 500 },
];

// Input dan Output
const xs = tf.tensor2d(
  data.map((item) => [
    item.usia,
    item.jenisKelamin,
    item.pendidikan,
    item.pengeluaran,
    item.keluarga,
    item.lokasi,
  ])
); // Input Data
const ys = tf.tensor2d(data.map((item) => [item.pendapatan])); // Target Data

// Membuat Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, inputShape: [6], activation: 'relu' })); // Hidden Layer
model.add(tf.layers.dense({ units: 1 })); // Output Layer
model.compile({
  optimizer: tf.train.adam(),
  loss: 'meanSquaredError',
});

// Melatih Model
(async () => {
  await model.fit(xs, ys, { epochs: 200 });
  console.log('Model telah dilatih.');

  // Prediksi Data Baru (Input Mentah)
  const input = tf.tensor2d([[30, 1, 4, 7, 3, 1]]); // Data Baru: Usia 30, Pria, Sarjana, Pengeluaran 7 juta, Keluarga 3, Lokasi 1
  const prediction = model.predict(input);
  prediction.print();

  // Hasil Prediksi
  console.log(`Pendapatan Tahunan yang Diprediksi: ${prediction.dataSync()[0]} juta`);
})();
