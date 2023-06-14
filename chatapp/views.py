from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import JsonResponse
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv, find_dotenv
import openai
ENV_FILE = find_dotenv()

if ENV_FILE:
    load_dotenv(ENV_FILE)

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')


aa= pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

class Newchat(TemplateView):
    template_name ='chat.html'

    def get(self,request):
        return render(request, self.template_name)

    def post(self,request):
        query = request.POST['prompt']
        embeddings = OpenAIEmbeddings()
        docsearch = Pinecone.from_existing_index(index_name="test-index", embedding=embeddings)

        llm = ChatOpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        docs = docsearch.similarity_search(query, top_k=1, include_metadata=True, metadata={'Meta': 'Flex GT'})
        if len(docs) > 0:
            doc = docs[0]  # Get the first document
            print(doc.page_content)
            print(doc.metadata)
            res = chain.run(input_documents=docs, question=query)

            context = {
                'response_by_AI': doc.page_content + str(doc.metadata),
                # 'response_by_AI':res,
            }
            print(context)
            return JsonResponse(context)



class StoreVectors(TemplateView):

    def get(self,request):
        loader = DirectoryLoader('./txt_data/', glob="./*.txt", loader_cls=TextLoader)
        data = loader.load()
        data[0].metadata["Meta"] = "Back Support AC"
        data[1].metadata["Meta"] = "Back Support HD"
        data[2].metadata["Meta"] = "Flex CD"
        data[3].metadata["Meta"] = "Flex GT"
        data[4].metadata["Meta"] = "Flex Heat"
        data[5].metadata["Meta"] = "Flex MLT"
        data[6].metadata["Meta"] = "Flex NP"
        data[7].metadata["Meta"] = "Flex SC"
        data[8].metadata["Meta"] = "Knee & Ankle CR"

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()

        pinecone.init(
            api_key='7c9b5573-82b0-49ed-acff-9f37fce7f803',
            environment='us-west1-gcp-free'
        )
        metadatas = []
        for text in texts:
            metadatas.append({
                "Meta": text.metadata["Meta"]
            })
        # print(metadatas)
        Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name="test-index",
                            metadatas=metadatas)
        return JsonResponse({'success':'success'})