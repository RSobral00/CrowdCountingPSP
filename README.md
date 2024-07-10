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


**Necessário:** Possuir as imagens e respetivos density maps de treino e de validação do DroneCrowd.

**Retorna**: Modelos treinados inicialmente no DroneCrowd e histórico de treino.

**Nota:** Foi feita a normalização dos density maps com o valor 0.042 que correspondia ao valor máximo de todos os density maps deste dataset (obtido com DC_Max_Value_DensityMap.py), ao fazer inferência em novas imagens é necessário multiplicar o resultado da soma dos píxeis por este valor para ser obtido a contagem correta de pessoas. Remover este passo durante o treino é sugerido para a próxima implementação.





## Código para avaliar os modelos no conjunto de teste DroneCrowd após serem inicialmente treinados:
-------------------------------------------------------------------------------------------------

* **DC_Model_Testing.py**
  

**Necessário:** Modelo treinado no DroneCrowd, imagens de teste e test_info.txt (csv gerado c/ informação da contagem de pessoas das imagens de teste).

Carrega o modelo e após isso é feita a inferência nas imagens de teste. 
	Este código também usa a informação verdadeira obtida na geração dos density maps "DroneCrowd/test_data/test_info.txt" 
	Resulta num csv em que cada linha corresponde à imagem, contagem verdadeira e contagem prevista. 

**Retorna**: Ficheiro csv c/ informação relativa a ["Img","N_Gt_Int","N_Gt",N_pred] que será usada na MSE_MAE_Calculator.py.

RMSE_MAE_Calculator.py:
	O csv que resulta do script anterior é carregado neste ficheiro de forma a calcular o RMSE e MAE 








Fine-Tuning c/ dados PSP:

---------------------------------------------------------
Os dados foram anotados no labelbox. 

Conjunto de Treino e Validação separados após gerar os density maps 75% e 25% 

Cenários de teste: C3,C4,C5,C13,C14,C15, C20_1,C20_6.

Imagens resized para 1080x1920.




PSP_Train_Val_Generator.py: 
---------------------------

Gerar dados de treino da PSP.

	Para os cenários de treino: Para os cenários C1,C2,C3 as coordenadas estão na resolução 1080x1920 (H,W), para os restantes estão em 2280x4056 e foram redimensionadas para 	1080x1920.





	Ler os ficheiros de anotações GT: Ex: C1_GT_1080.ndjson:
--------------------------------------------------------------------
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









Fine tuning com conjunto de treino da PSP após terem sido treinados no DroneCrowd:
------------------------------------------------------------------------------------	


PSP_DC_ARCN.py; PSP_DC_CSRNET.py; PSP_DC_SoftCSRNetP.py;

Funcionamento semelhante aos ficheiros de treino dos modelos para o DroneCrowd.




Obter os resultados para o conjunto de teste da PSP: 
-----------------------------------------------------
PSP_Model_Inference.py

Cada mapa de densidade previsto da respetiva imagem de teste é guardado de forma a ser possível obter métricas de avaliação mais tarde. 

Informação dos valores reais das contagens está em -> PSP_GT_Info.txt
