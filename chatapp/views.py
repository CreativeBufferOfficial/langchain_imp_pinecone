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
# pinecone.delete_index("cb-index")
# index_name = "cb-index"
# pinecone.create_index(index_name, dimension=1536, metric="cosine", pod_type="p1.x1")

folder_path = 'Kaven_langchain/txt_data'
def get_file_list():
    file_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_list.append(filename.replace('.txt', ''))
    return file_list

# file_list = get_file_list(folder_path)
# print(len(file_list))



class Newchat(TemplateView):
    template_name ='chat.html'

    def get(self,request):
        file_list = get_file_list()
        return render(request, self.template_name, {'file_names': file_list})

    def post(self,request):
        query = request.POST['prompt']
        selected_file=request.POST['selected_file']
        print(selected_file)
        embeddings = OpenAIEmbeddings()
        docsearch = Pinecone.from_existing_index("cb-index",
                                                 embedding=OpenAIEmbeddings(),
                                                 namespace=selected_file)
        output = docsearch.similarity_search(query, k=1, return_metadata=True)
        print(output)
        page_contents = [o.page_content for o in output]
        print(page_contents)
        context = {
            'response_by_AI': page_contents,
            }
        print(context)
        return JsonResponse(context)



class StoreVectors(TemplateView):

    def get(self,request):
        # pinecone.delete_index("cb-index")
        # index_name = "cb-index"
        # pinecone.create_index(index_name, dimension=1536, metric="cosine", pod_type="p1.x1")
        # folder_path2 = './txt_data'

        loader = DirectoryLoader(folder_path, glob="./*.txt", loader_cls=TextLoader,loader_kwargs={'encoding': "utf-8"})
        data = loader.load()
        print(len(data))

        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)

        texts = []
        for i in data:
            text = text_splitter.split_documents([i])
            texts.append(text)
        print(len(texts))

        namespace_names = get_file_list()
        data_length = len(data)

        for i in range(data_length):
            metadatas = [{"page": j} for j in range(len(texts[i]))]
            Pinecone.from_texts(
                [t.page_content for t in texts[i]],
                embeddings,
                index_name="cb-index",
                metadatas=metadatas,
                namespace=namespace_names[i],
            )
        return JsonResponse({'success':'success'})