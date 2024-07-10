# CrowdCountingPSP
**Projeto de dissertação Rúben Sobral Universidade de Aveiro c/ PSP de Aveiro "Contagem de Pessoas Através de Imagens Recolhidas por um VANT em Operação"**
------------------------------------------------------------------------------------------------------------------------------------------------------------


# DroneCrowd

Download DroneCrowd (Full Version): https://github.com/VisDrone/DroneCrowd  


## Código para Gerar os Mapas de Densidade do DroneCrowd:
-------------------------------------------------------

* **DC_Train_Data_Generator.py**
* **DC_Val_Data_Generator.py** 
* **DC_Test_Data_Generator.py**
* DC_Create_DensityMap_Gaussian.py


Executar DC_Train_Data_Generator.py antes de DC_Val_Data_Generator.py.  DC_Test_Data_Generator.py quando desejado. 

**Necessário:**  **trainlist.txt** e **testlist.txt** pertencentes ao DroneCrowd, respetivas anotações das pessoas e imagens. 

**Retorna** os mapas de densidade gerados com o auxílio do ficheiro: DC_Create_DensityMap_Gaussian.py que é usado em **DC_Train_Data_Generator.py** e **DC_Test_Data_Generator.py**.




## Código de Treino dos Modelos ARCN, CSRNET, Soft-CSRNet+ no DroneCrowd:
-------------------------------------------------------------------------
* **DC_ARCN.py**
* **DC_CSRNET.py**
* **DC_SoftCSRNETP.py**


**Necessário:** Possuir as imagens e respetivos mapas de densidade de treino e de validação do DroneCrowd.

**Retorna**: Modelos treinados inicialmente no DroneCrowd e histórico de treino.

**1ª Nota:** Foi feita a normalização dos mapas de densidade com o valor 0.042 que correspondia ao valor máximo de todos os mapas de densidade deste dataset (obtido com DC_Max_Value_DensityMap.py), ao fazer inferência em novas imagens é necessário multiplicar o resultado da soma dos píxeis por este valor para ser obtido a contagem correta de pessoas. Remover este passo durante o treino é sugerido para a próxima implementação.

**2ª Nota:** É necessário possuir o modelo VGG_16.json e os respetivos pesos VGG_16.h5 obtidos em 
https://github.com/CS3244-AY2021-SEM-1/csrnet-tensorflow/tree/maste




## Código de Avaliação dos Modelos no Conjunto de Teste do DroneCrowd Após Treino:
-------------------------------------------------------------------------------------------------

* **DC_Model_Testing.py**
* **RMSE_MAE_Calculator.py**
  

**Necessário:** Modelo treinado no DroneCrowd, imagens de teste e test_info.txt (csv gerado c/ informação da contagem de pessoas das imagens de teste).

**Retorna**: Ficheiro csv c/ informação relativa a ["Img","N_Gt_Int","N_Gt",N_pred] que será usada na MSE_MAE_Calculator.py.

**Nota:** Após execução do DC_Model_Testing.py, carregar o csv resultante com a informação de teste verdadeira e obtida pela execução do DC_Model_Testing.py. Este ficheiro irá resultar num print com as métricas de erro RMSE e MAE normalmente usadas para comparar os modelos com outros trabalhos.
 		








## Fine-Tuning c/ dados PSP:
---------------------------------------------------------

Após os modelos terem sido inicialmente treinados no DroneCrowd, foi aplicado fine-tuning com o objetivo de melhorar os modelos para características semelhantes às de operação pela PSP. 

### PSP Dataset
--------------------------------------------------------
* Pessoas anotadas c/ o LabelBox
* Imagens redimensionadas para 1080x1920 (H,W).
* Para os cenários C1,C2,C3 as anotações (coordenadas de cada pessoa) estão na resolução 1080x1920 (H,W), para os restantes estão em 2280x4056 e é necessário converter para 1080x1920.
* Dataset que consiste em 374 imagens com 210 de treino e 164 de teste. 
* Nomenclatura das imagens CXX_HYY_AZZ.JPG: CXX - Cenário XX, HYY: Altura YY, AZZ: Ângulo ZZ. 
* As imagens que não possuem H e A não têm altitude nem ângulo anotados mas têm características das imagens de operação da PSP.
* **Conjunto de Teste:** Consiste nas imagens dos cenários C3,C4,C5,C13,C14,C15, C20_1,C20_6.
* **Conjunto de Treino** e **Validação** separados após gerar os mapas de densidade 75% e 25% do conjunto de treino total.
* Ficheiro de anotação diferente para cada cenário.


**Obtenção das coordenadas de cada pessoa a partir do ficheiro das anotações de cada cenário:** 

	gt_file = "C1_GT_1080.ndjson"

	gt_full_path = os.path.join(gt_annotation_path,gt_file)
 	
	with open(gt_full_path, "r") as file:

        content = file.read()

        # Parse the JSON content
        data = [json.loads(line) for line in content.split("\n") if line.strip()]


        for entry in data:

            external_id = entry["data_row"]["external_id"]     #ExternalID é o nome de cada imagem 


            image_path = os.path.join(images_path, external_id)

            if os.path.exists(image_path):
                
                print(f"Image {external_id}: Generating.")


                coordinates = []

                for annotation in entry["projects"]["clrzfkqls00yi07wp3ndhb7nx"]["labels"][0]["annotations"]["objects"]:
                    if annotation["name"] == "People":
                        point = annotation["point"]
                        coordinates.append((point["y"],point["x"]))




### Gerar Dados de Treino PSP Dataset:
--------------------------------------------------------
* **PSP_Train_Val_Generator.py**

Executar o ficheiro PSP_Train_Val_Generator.py.

**Necessário:**: Ficheiros de anotações em formato .ndjson disponibilizados no dataset da PSP. 

**Retorna:** Mapas de densidade de treino e de validação. Alterar os caminhos das pastas para cada de forma a gerar um em cada execução.




### Treinar os Modelos Pre-treinados no DroneCrowd no dataset da PSP (FT)
--------------------------------------------------------

* PSP_DC_ARCN.py
* PSP_DC_CSRNET.py
* PSP_DC_SoftCSRNetP.py

Funcionamento semelhante aos ficheiros de treino dos modelos para o DroneCrowd.

**Necessário:** Modelos treinados no DroneCrowd e imagens com correspondentes mapas de densidade de treino e validação do dataset da PSP.

**Retorna:** Os modelos após fine-tuning no conjunto de dados da PSP. 



### Obter os resultados para o conjunto de teste da PSP: 
--------------------------------------------------------
* **PSP_Model_Inference.py**

**Necessário:** Um dos modelos treinados, tanto sem fine-tuning ou com fine-tuning e um caminho específico para guardar as estimativas e PSP_GT_Info.txt que contém a informação real.

**Retorna:** Os mapas de densidade estimados do conjunto de teste da PSP pelo modelo escolhido.


* **Win_PSP_Test.py:** Usado para obter métricas de erro MAE, RMSE, MAPE
  
* **Win_SSIM.py:** Usado para obter MSSIM dos mapas de densidade estimados e reais, necessário os mapas de desidade do conjunto de teste estarem gerados. 

