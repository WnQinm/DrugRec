from src.retriever.bing_retriever import BingRetriever
r = BingRetriever("./checkpoint/facebook-contriever", "./checkpoint/retriever-pretrained-checkpoint")
print(r.query("What is Amoxicillin?"))
