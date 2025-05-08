const fs = require('fs'); const data = JSON.parse(fs.readFileSync('data/qa_en_dataset_fixed_domains.json')); data.train_data.forEach((d, i) => console.log());
