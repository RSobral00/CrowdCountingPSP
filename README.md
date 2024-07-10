# CrowdCountingPSP
Projeto de dissertação Rúben Sobral Universidade de Aveiro c/ PSP de Aveiro "Contagem de Pessoas Através de Imagens Recolhidas por um VANT em Operação"
------------------------------------------------------------------------------------------------------------------------------------------------------------
Código para gerar os mapas de densidade do DroneCrowd:
-------------------------------------------------------

DC_Train_Data_Generator.py: 
	Gera os mapas de densidade de treino do DroneCrowd.
	Abre a lista que contem informação sobre as sequências que são de treino. 
	Para cada sequência: 
		Recolhe as coordenadas de cada pessoa em cada frame. 
		Após isso, para cada frame, é gerado o mapa de densidade com o uso do script DC_Create_DensityMap_Gaussian.py
	No fim, guarda informação relativa a cada frame, neste caso, o número de pessoas. 



DC_Test_Data_Generator.py:
	Mesmo que o DC_Train_Data_Generator.py mas para os dados de teste do DroneCrowd. 


DC_Val_Data_Generator.py:
	Copia os density maps de teste já gerados que na realidade são de validação para as pastas apropriadas.
	

DC_Create_DensityMap_Gaussian:
	Gera o mapa de densidade ao convolver o filtro gaussiano para cada ponto c/ valor 1 da imagem binária (coordenadas do centro da cabeça).




Código para treinar os modelos ARCN, CSRNET, Soft-CSRNet+ no DroneCrowd:
-------------------------------------------------------------------------
	DC_ARCN.py; DC_CSRNET.py; DC_SoftCSRNETP.py
		
	Após geração dos mapas de densidade para os três conjuntos a partir destes scripts é possível treinar os três modelos para o DroneCrowd, onde os modelos treinados são guardados 
	na pasta "Results" com o histórico de treino. 
	
	Foi feita a normalização dos density maps c o valor 0.042 que correspondia ao valor máximo de todos os density maps deste dataset, ao fazer inferência em novas imagens
	é necessário multiplicar o resultado da soma dos píxeis por este valor para ser obtido a contagem correta de pessoas. Remover este passo durante o treino é sugerido para a próxima.




Código para avaliar os modelos no conjunto de teste DroneCrowd após serem inicialmente treinados:
-------------------------------------------------------------------------------------------------

DC_Model_Testing.py:
	Carrega o modelo e após isso é feita a inferência nas imagens de teste. 
	Este código também usa a informação verdadeira obtida na geração dos density maps "DroneCrowd/test_data/test_info.txt" 
	Resulta num csv em que cada linha corresponde à imagem, contagem verdadeira e contagem prevista. 



RMSE_MAE_Calculator.py:
	O csv que resulta do script anterior é carregado neste ficheiro de forma a calcular o RMSE e MAE 






-----------------------------------------------------------

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
