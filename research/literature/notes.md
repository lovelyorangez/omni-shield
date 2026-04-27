# Literature Review Notes

## Papers to read and cite

### PII Detection Systems
- [ ] Microsoft Presidio (2020) github.com/microsoft/presidio
- [ ] AWS Comprehend Detect PII — Amazon docs
- [ ] Google Cloud DLP — Google docs
- [ ] spaCy NER — Honnibal et al.
- [ ] "A Survey on Automated PII Detection" — search arxiv

### VLM Document Understanding
- [ ] Qwen-VL (Bai et al., 2023) — base model we fine-tuned
- [ ] DocVQA (Mathew et al., 2021)
- [ ] LayoutLM (Xu et al., 2020)
- [ ] Donut (Kim et al., 2022)

### Knowledge Distillation
- [ ] "Distilling the Knowledge in a Neural Network" (Hinton 2015)
- [ ] LoRA (Hu et al., 2021)
- [ ] QLoRA (Dettmers et al., 2023)

### Edge ML Deployment
- [ ] bitsandbytes quantization paper
- [ ] "LLM in a Flash" (Apple, 2023)

### Blockchain Audit Trails
- [ ] Ethereum whitepaper (Buterin, 2014)
- [ ] "Blockchain for Data Auditing" — survey

### ZK-SNARKs / Privacy
- [ ] Groth16 (Groth, 2016) — proof system we use
- [ ] ZoKrates (Eberhardt & Tai, 2018)
- [ ] "Succinct Non-Interactive Arguments" survey

### Multi-Agent Systems
- [ ] AutoGen (Wu et al., 2023)
- [ ] MetaGPT (Hong et al., 2023)

---

## Novelty gap table

| System      | Local | Multi-modal | Blockchain | ZK Proof | VLM fine-tuned |
|-------------|-------|-------------|------------|----------|----------------|
| Presidio    | Yes   | No          | No         | No       | No             |
| AWS DLP     | No    | Partial     | No         | No       | No             |
| Google DLP  | No    | Partial     | No         | No       | No             |
| Omni-Shield | Yes   | Yes (6)     | Yes        | Yes      | Yes            |

---

## Key claims to substantiate
1. First local-first multi-modal PII redaction system
2. First to combine VLM fine-tuning + blockchain + ZK-SNARK
3. Runs on consumer 8GB VRAM GPU
4. Knowledge distillation with synthetic data for document PII

After the literature review, confirm these claims hold
by checking if any paper combines all four properties.
