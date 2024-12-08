We have worked on the Brainteaser-QA task on the SEMEVAL dataset for semantic and contextual reconstruction of puzzle based questions. We measured group based and instance based accuracies of the answers to the questions by applying several Large Language models.
Then we had to fine-tune the LLMs to suit this dataset. To this end, we used LoRA and PEFT to reduce the number of trainable parameters while fine-tuning. After fine-tuning, the accuracy scores improved significantly for the LLMs. I have worked mainly with Llama 3.2 with 
3B parameters, Phi1-5 and GPT 2 model fine-tuning and measuring how the group based accuracies relate to the cosine similarity between the questions. We are grateful to the CS department in the University of Cincinnati for providing us Nvidia 6000 ADA GPUs which 
proved to be extremely useful for fine-tuning LLM models. More details on the LLM models performance, hyper parameter tuning, loss divergence, best model selection and conclusive analysis are available on the powerpoint titled "NLP Final Presentation.pptx"
and "Final_Report.docx"
