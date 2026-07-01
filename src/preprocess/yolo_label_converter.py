import os
import argparse
import concurrent.futures
from pathlib import Path
from functools import partial

def _processar_arquivo_unico(txt_file, output_path, classes_map, img_width, img_height, norm_width, norm_height):
    """
    Função auxiliar executada em paralelo para converter um único arquivo.
    """
    linhas_yolo = []
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
            
        for linha in linhas:
            partes = linha.strip().split()
            if len(partes) != 3:
                continue
                
            classe_nome, x_str, y_str = partes
            
            if classe_nome not in classes_map:
                continue
                
            class_id = classes_map[classe_nome]
            x_center_abs = float(x_str)
            y_center_abs = float(y_str)
            
            # Normalização YOLO
            x_center_norm = x_center_abs / img_width
            y_center_norm = y_center_abs / img_height
            
            linha_yolo = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {norm_width:.6f} {norm_height:.6f}"
            linhas_yolo.append(linha_yolo)
            
        # Salva o resultado na pasta de destino
        output_file = output_path / txt_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(linhas_yolo))
            
        return True
    except Exception as e:
        print(f"[-] Erro ao processar o arquivo {txt_file.name}: {e}")
        return False

def converter_para_yolo_paralelo(input_dir, output_dir, img_width, img_height, bbox_size=50, max_workers=None):
    """
    Lê arquivos .txt de anotação e os transforma para o formato YOLO em paralelo.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    classes_map = {
        "mitose": "0",
        "nao_mitose": "1"
    }
    
    norm_width = bbox_size / img_width
    norm_height = bbox_size / img_height
    
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"[-] Nenhum arquivo .txt encontrado em: {input_path.resolve()}")
        return

    print(f"[+] Iniciando conversão paralela de {len(txt_files)} arquivos usando múltiplos núcleos...")

    # Congela os argumentos fixos para enviar os arquivos mapeados no Executor
    func_auxiliar = partial(
        _processar_arquivo_unico, 
        output_path=output_path, 
        classes_map=classes_map, 
        img_width=img_width, 
        img_height=img_height, 
        norm_width=norm_width, 
        norm_height=norm_height
    )

    # Dispara os processos em paralelo
    sucessos = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # O executor.map distribui a lista de arquivos entre os workers
        resultados = executor.map(func_auxiliar, txt_files)
        sucessos = sum(1 for res in resultados if res)
            
    print(f"[+] Sucesso! {sucessos}/{len(txt_files)} arquivos processados e salvos em: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Absolute Annotations to YOLO Format in Parallel.")
    
    parser.add_argument('-i', '--input', type=str, required=True, help="Diretório dos arquivos TXT originais")
    parser.add_argument('-o', '--output', type=str, required=True, help="Diretório onde os arquivos YOLO serão salvos")
    parser.add_argument('-w', '--width', type=int, required=True, help="Largura fixa das imagens (pixels)")
    parser.add_argument('-hgt', '--height', type=int, required=True, help="Altura fixa das imagens (pixels)")
    parser.add_argument('-b', '--bbox', type=int, default=50, help="Tamanho da Bounding Box (padrão: 50)")
    parser.add_argument('--workers', type=int, default=None, help="Número de cores CPU (padrão: usar todos)")

    args = parser.parse_args()
    
    converter_para_yolo_paralelo(
        input_dir=args.input, 
        output_dir=args.output, 
        img_width=args.width, 
        img_height=args.height, 
        bbox_size=args.bbox,
        max_workers=args.workers
    )