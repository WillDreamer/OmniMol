from transformers import PreTrainedTokenizer
from data_pipe.chat_template import apply_chat_template
from data_pipe.chat_template import tokenizer_image_token
from torch.utils.data import Dataset, ConcatDataset, random_split
import json
import selfies
import torch
import pandas as pd
import random
import os
from data_pipe.mol_utils import smiles2graph
from loggers import WrappedLogger
from collections import OrderedDict
from data_pipe.data_utils import GraphDatasetCollator
from data_pipe import conversation_lib
import re
import selfies as sf
from rdkit import Chem
logger = WrappedLogger(__name__)

def check_output(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: PreTrainedTokenizer
):
    ids = list(input_ids.detach().numpy())
    logger.info(f"ids: {ids}")
    logger.info(f"labels {labels}")
    ids.remove(-200)
    logger.info([tokenizer.decode(ids)])
    

def apply_prompt(message):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    return prompt


def convert_selfies_to_smiles(input_str):
    def replace_selfies(match):
        # 提取 SELFIES 字符串
        selfies_str = match.group(1)
        # 将 SELFIES 转换为 SMILES
        smiles = sf.decoder(selfies_str)
        if smiles:
            # 将 SMILES 转换为 RDKit 分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 生成 canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                return f'[START_SMILES]{canonical_smiles}[END_SMILES]'
        # 如果转换失败，保持原样
        return match.group(0)
    
    # 定义正则表达式模式，匹配 [START_SELFIES] 和 [END_SELFIES] 之间的内容
    pattern = r'\[START_SELFIES\](.*?)\[END_SELFIES\]'
    # 替换所有匹配项
    modified_str = re.sub(pattern, replace_selfies, input_str)
    return modified_str

class MetaGraphDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        for_test: bool,
        task_name: str,
        data_file: list[dict] = None,
        if_smiles: bool = False
    ) -> None:
        super().__init__()
        if data_file is None:
            with open(data_path, "rb") as f:
                self.list_data_dict = json.load(f)
                f.close()
        else:
            self.list_data_dict = data_file

        self.tokenizer = tokenizer
        self.task_name = task_name
        self.if_smiles = if_smiles
        logger.info(f"Task \033[34m{task_name}\033[0m\t Total number of samples: {self.__len__()}", on_rank0=True)
        if for_test:
            self.filter_for_test()
            logger.info(f"Filtered {self.__len__()} for test", on_rank0=True)
        else:
            self.filter_for_training()
            logger.info(f"Filtered {self.__len__()} for training", on_rank0=True)
        
    def selfies2smiles(self, selfies_str: str) -> str | None:
    
        # 第一步：SELFIES 转为 SMILES
        smiles = sf.decoder(selfies_str)
        if not smiles:  # 检查是否解码失败
            return None
        # 第二步：SMILES 转为 canonical SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # 检查 RDKit 是否解析失败
            return smiles
        return Chem.MolToSmiles(mol, canonical=True)  # 返回 canonical SMILES
            

    def filter_for_training(self) -> None:
        self.list_data_dict = [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'train']
    
    def filter_for_test(self) -> None:
        self.list_data_dict =  [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'test']
        
    def _yield_prompt(self, instruction, graphs, gt):
        prompt = apply_prompt(instruction)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
    
        data = {
            "input_ids": input_ids,
            "graphs": graphs,
            "gt": gt,
            "prompt": prompt,
            "this_task_ids": torch.LongTensor([0])
        }
        
        return data
        
    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        pass
    
    
class PretrainMolDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_file: list[dict] = None,
        **kwargs
    ) -> None:
        super().__init__()
        if data_file is None:
            list_data_dict = pd.read_csv(data_path)
            self.list_data_dict = list_data_dict
        else:
            self.list_data_dict = data_file
            
        self.tokenizer = tokenizer
        
        logger.info(f"Total number of samples: {self.__len__()}", on_rank0=True)
        print("====Pretrain Molecule Description Dataset====")
        
        self.question_pool = [
        'Could you give me a brief overview of this molecule?',
        'Could you provide a description of this molecule?',
        'Describe this molecule.',
        'Please give me some details about this molecule.',
        'Provide a brief overview of this molecule.',
        'Provide a description of this molecule.',
        'What can you tell me about this molecule?'
        ]
        
        
    def __len__(self):
        return len(self.list_data_dict)
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        smiles, description = self.list_data_dict["SMILES"][i], self.list_data_dict["Description"][i]
        
        instruction = random.choice(self.question_pool)
        instruction = "<image>\n" + instruction
        
        message = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": description}
            ]
        ]
        
        graph_for_molecule = smiles2graph(smiles)
        assert graph_for_molecule is not None, f"Found molecule that cannot be converted to graph: {smiles}"
        
        data_dict = apply_chat_template(message, self.tokenizer, graph_for_molecule is not None)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        data_dict['graphs'] = graph_for_molecule
        assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"

        return data_dict


class ForwardPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
        ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Forward Prediction==",
            data_file,
            **kargs
        )
        self.add_selfies = add_selfies
        self.for_test = for_test
                
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # 1. Get sample
        raw = self.list_data_dict[i]

        # 2. Get instruction, input selfies, output selfies
        instruction = raw['instruction']
        inputs, output_selfies = raw['input'].split('.'), raw['output']
        if self.if_smiles:
            inputs_mol = self.selfies2smiles(raw['input'])
            output_mol = self.selfies2smiles(raw['output'])  
        else:
            inputs_mol = raw['input']
            output_mol = raw['output']
        
        # 3. Convert to Graph
        reactant_smiles = self.selfies2smiles(inputs[0])
        graph_for_first_reactant = smiles2graph(reactant_smiles)

        # 4. Add SELFIES
        if self.add_selfies:
            instruction += " " + inputs_mol
        elif len(inputs) > 1:
            instruction += f" The other joint reactants are: {','.join(inputs[1:])}"
            
        instruction = "<image>\n" + instruction
        
        # test routine
        if self.for_test:
            return self._yield_prompt(instruction, graph_for_first_reactant, output_mol)
        
        # 5. Prepare conversations
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_mol}
            ]
        ]

        # Tokenization
        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph_for_first_reactant is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        assert graph_for_first_reactant is not None, f"Cannot convert {inputs[0]} to graph"
        data_dict['graphs'] = graph_for_first_reactant
        assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"
        data_dict["this_task_ids"] = torch.LongTensor([0])

        return data_dict
    
    
class ReagentPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Reagent Prediction==",
            data_file,
            **kargs
        )
        
        self.add_selfies = add_selfies
        self.for_test = for_test
        
    @staticmethod
    def construct_instruct_question(product:str):
        """
        Construct instruct question for each graph
        """
        question_pools = [
            'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
            'Give some possible reagents that could have been used in the following chemical reaction.',
            'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
            'Please provide possible reagents based on the following chemical reaction.',
        ]
        question = random.choice(question_pools)
        question += f"\nThe product is {product}"
        return question    

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        if self.if_smiles:
            reactant_mol = self.selfies2smiles(reactant)
            product_mol = self.selfies2smiles(product)
            output_mol = self.selfies2smiles(raw['output'])
        else:
            reactant_mol = reactant
            product_mol = product
            output_mol = raw['output']
            
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if not self.add_selfies:
            # insert product to the instruction end
            instruction = self.construct_instruct_question(product_mol)
        else:
            instruction = raw['instruction'] + f" The reaction is {'>>'.join([reactant_mol,product_mol])}"

        instruction = "<image>\n" + instruction
        
        graph=smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_mol)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_mol}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])


        assert graph is not None, f"Cannot convert {reactant} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([1])
            
        return data_dict
    
    
class RetrosynDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Retrosynthesis==",
            data_file,
            **kargs
        )
        self.add_selfies = add_selfies
        self.for_test = for_test
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.if_smiles:
            inputs_mol = self.selfies2smiles(raw['input'])
            output_mol = self.selfies2smiles(raw['output'])  
        else:
            inputs_mol = raw['input']
            output_mol = raw['output']
        if self.add_selfies:
            instruction += f" The product is: {inputs_mol}"
        

        instruction = "<image>\n" + instruction
        
        input_selfies, output_selfies = raw['input'], raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(input_selfies)
        
        graph=smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_mol)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_mol}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input_selfies} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([2])
        
        return data_dict
    
    
class PropertyPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==HOMO LUMO==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        if self.add_selfies:
            instruction += f" The compound sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        
        input_selfies, target = raw['input'], str(raw['output'])
        graph=smiles2graph(self.selfies2smiles(input_selfies))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, target)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": target}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input_selfies} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([3])
        
        return data_dict
    
class MolcapDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Molcap==",
            data_file,
            **kargs
        )
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        self.for_test = for_test
        self.add_selfies = add_selfies
        
    def maybe_drop_selfies(
        self,
        data_dict,
        messages,
        has_image
    ):
        if len(data_dict['input_ids']) > self.tokenizer.model_max_length:
            logger.warning(f"input too long {len(data_dict['input_ids'])}, selfies dropped.")
            instruction = random.choice(self.question_pool)
            instruction = "<image>\n" + instruction
            messages[0][0]["value"] = instruction
            data_dict = apply_chat_template(messages, self.tokenizer, has_image=has_image)
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            logger.warning(f"Adjusted length {len(data_dict['input_ids'])}")
            
        return data_dict
        
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = random.choice(self.question_pool)
        # instruction = "What is the name of this molecule?"
        input = raw['input']
        output = raw['output']
        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        
        if self.add_selfies:
            instruction += f" The compound sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        
        graph = smiles2graph(self.selfies2smiles(input))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        data_dict = self.maybe_drop_selfies(data_dict, messages, has_image=(graph is not None))
        
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([4])

        return data_dict
    
    
class CatalystPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Catalyst Prediction dataset==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        if self.if_smiles:
            reactant_mol = self.selfies2smiles(reactant)
            product_mol = self.selfies2smiles(product)
            output_mol = self.selfies2smiles(raw['output'])
        else:
            reactant_mol = reactant
            product_mol = product
            output_mol = raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {'>>'.join([reactant_mol,product_mol])}."
        else:
            instruction = raw['instruction']

        instruction = "<image>\n" + instruction
        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_mol)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_mol}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([5])

        return data_dict


class SolventPredDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Solvent Prediction dataset==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        if self.if_smiles:
            reactant_mol = self.selfies2smiles(reactant)
            product_mol = self.selfies2smiles(product)
            output_mol = self.selfies2smiles(raw['output'])
        else:
            reactant_mol = reactant
            product_mol = product
            output_mol = raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {'>>'.join([reactant_mol,product_mol])}."
        else:
            instruction = raw['instruction']
        # elif len(input) > 1:
        #     instruction = ""
        #     instruction += f" The other joint reactants are: {','.join(input[1:])}"

        instruction = "<image>\n" + instruction

        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_mol)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_mol}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([6])

        return data_dict


class YieldRegressionDataset(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Yield Regression dataset==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        if self.if_smiles:
            reactant_mol = self.selfies2smiles(reactant)
            product_mol = self.selfies2smiles(product)
        else:
            reactant_mol = reactant
            product_mol = product

        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {'>>'.join([reactant_mol,product_mol])}."
        else:
            instruction = raw['instruction']

        instruction = "<image>\n" + instruction
        graph = smiles2graph(reactant_smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": str(output_selfies)}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([7])

        return data_dict
    
class ExpProcedurePrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Experimental Procedure Prediction dataset==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # 取出数据条目
        raw = self.list_data_dict[i]

        # 从raw中提取extracted_molecules字典
        extracted_molecules = raw.get("extracted_molecules", {})

        # extracted_molecules是 {SMILES: "$1$"} 的形式，需要反转成 {"$1$": SMILES} 方便查询
        placeholder_to_smiles = {placeholder: smi for smi, placeholder in extracted_molecules.items()}

        # 从raw["input"]中查找所有占位符（例如$1$, $2$, $-1$等）
        placeholders = re.findall(r"\$\d+\$", raw["input"])

        # 收集匹配到的所有SMILES
        smiles_list = []
        for ph in placeholders:
            # 如果该占位符在placeholder_to_smiles中，则取出对应的SMILES
            if ph in placeholder_to_smiles:
                smiles_list.append(placeholder_to_smiles[ph])

        # 将所有SMILES用"."连接，形成一个字符串
        smiles = ".".join(smiles_list)

        if self.if_smiles:
            # instruction = "Reactants: "
            # for idx, k in enumerate(placeholder_to_smiles):
            #     if idx == len(placeholder_to_smiles) - 1:
            #         idx_ = -1
            #         instruction += f"${idx_}$: [START_SMILES]{placeholder_to_smiles[f"${idx_}$"]}[END_SMILES]"
            #     else:
            #         instruction += f"${idx}$: [START_SMILES]{placeholder_to_smiles[f"${idx}$"]}[END_SMILES]"
            input, output_selfies = convert_selfies_to_smiles(raw['input']), raw['output']
            
        else:
            # 使用raw["input"]和raw["output"]构造instruction
            input, output_selfies = raw['input'], raw['output']
        
        instruction = raw['instruction'] + f"{input}. "
        instruction += "The Action Sequence: "
        instruction = "<image>\n" + instruction

        # 根据data_args决定是用himol还是smiles2graph构造graph
        assert smiles is not None, f"Found invalid data {raw}"
        graph = smiles2graph(smiles)
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output_selfies)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([8])

        return data_dict
    
    
class SCFPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==SCF Prediction==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "SCF Energy"]
        logger.info(f"The actual length for SCF is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]

        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']

        if self.add_selfies:
            instruction += f" The molecule sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([9])

        return data_dict
    
    
class LogPPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==LogP Prediction==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "LogP"]
        logger.info(f"The actual length for LogP is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]

        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        
        if self.add_selfies:
            instruction += f" The molecule sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([10])

        return data_dict
    

class DescriptionQA(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Description QA==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "Description"]
        logger.info(f"The actual length for Description QA is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]

        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        
        if self.add_selfies:
            instruction += f" The compound sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([11])

        return data_dict
    
class WeightPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Weight Prediction==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "Molecular Weight"]
        logger.info(f"The actual length for Weight Prediction is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        output = raw["output"]

        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        
        if self.add_selfies:
            instruction += f" The molecule sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(mol))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([12])

        return data_dict
    
class TPSAPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Topological Polar Surface Area==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "Topological Polar Surface Area"]
        logger.info(f"The actual length for Topological Polar Surface Area is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]

        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']
        
        if self.add_selfies:
            instruction += f" The compound sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([13])

        return data_dict
    
    
class ComlexityPrediction(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None,
        **kargs
    ):
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==Complexity==",
            data_file,
            **kargs
        )
        self.for_test = for_test
        self.add_selfies = add_selfies
        self.list_data_dict = [raw for raw in self.list_data_dict if raw["metadata"]["task"] == "Complexity"]
        logger.info(f"The actual length for complexity is {len(self.list_data_dict)}", on_rank0=True)
        
    def __getitem__(self, i):
        raw = self.list_data_dict[i]
        instruction = raw["instruction"]
        mol = raw["input"]
        to_graph = mol.split(".")[0]
        output = raw["output"]
        
        if self.if_smiles:
            input_mol = self.selfies2smiles(raw['input'])
        else:
            input_mol = raw['input']

        if self.add_selfies:
            instruction += f" The compound sequence is: {input_mol}"

        instruction = "<image>\n" + instruction
        graph = smiles2graph(self.selfies2smiles(to_graph))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        assert graph is not None, f"Cannot convert {mol} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([14])

        return data_dict
    
class IUPACID(MetaGraphDataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        add_selfies: bool,
        for_test: bool,
        data_file: list[dict] = None
    ) -> None:
        super().__init__(
            data_path,
            tokenizer,
            for_test,
            "==IUPAC==",
            data_file
        )
        
        self.for_test = for_test
        self.add_selfies = add_selfies
        
    def maybe_drop_selfies(
        self,
        data_dict,
        messages,
        has_image
    ):
        if len(data_dict['input_ids']) > self.tokenizer.model_max_length:
            logger.warning(f"input too long {len(data_dict['input_ids'])}, selfies dropped.")
            instruction = random.choice(self.question_pool)
            instruction = "<image>\n" + instruction
            messages[0][0]["value"] = instruction
            data_dict = apply_chat_template(messages, self.tokenizer, has_image=has_image)
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            logger.warning(f"Adjusted length {len(data_dict['input_ids'])}")
            
        return data_dict
        
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        # instruction = "What is the name of this molecule?"
        input = raw['input']
        output = raw['output']
        
        if self.add_selfies:
            instruction += f" The compound sequence is: {input}"

        instruction = "<image>\n" + instruction
        
        graph = smiles2graph(self.selfies2smiles(input))
        
        if self.for_test:
            return self._yield_prompt(instruction, graph, output)
        
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        
        data_dict = self.maybe_drop_selfies(data_dict, messages, has_image=(graph is not None))
        
        assert graph is not None, f"Cannot convert {input} to graph"
        data_dict['graphs'] = graph
        assert -200 in data_dict["input_ids"]
        data_dict["this_task_ids"] = torch.LongTensor([4])

        return data_dict

NAME2DATASET = {
    "pubchem": PretrainMolDataset,
    "forward": ForwardPredDataset,
    "reagent": ReagentPredDataset,
    "retrosynthesis": RetrosynDataset,
    "molcap": MolcapDataset,
    "homolumo": PropertyPredDataset,
    "solvent": SolventPredDataset,
    "catalyst": CatalystPredDataset,
    "yield": YieldRegressionDataset,
    "experiment": ExpProcedurePrediction,
    "scf": SCFPrediction,
    "complexity": ComlexityPrediction,
    "tpsa": TPSAPrediction,
    "weight": WeightPrediction,
    "dqa": DescriptionQA,
    "logp": LogPPrediction,
    "iupac": IUPACID
}

MAP2FILEMAME = {
    "pubchem": "pubchem",
    "forward": "forward",
    "reagent": "reagent",
    "retrosynthesis": "retrosynthesis",
    "molcap": "molcap_train",
    "homolumo": "property",
    "solvent": "solvent",
    "catalyst": "catalyst",
    "yield": "yields_regression",
    "experiment": "exp_procedure",
    "scf": "3d_moit",
    "complexity": "3d_moit",
    "tpsa": "3d_moit",
    "weight": "3d_moit",
    "dqa": "3d_moit",
    "logp": "3d_moit",
    "iupac" : "selfies2iupac"
}

def build_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    for_test: bool,
    add_selfies: bool,
    split_val: bool,
    val_ratio: float,
    task_config: str = None, 
    sample_from_ratio: str = None,
    total_size: int = None,
    if_smiles: bool = False,
) -> Dataset:
    # task_name or sample_from_ratio: forward:0.3/reagent:0.5/...
    # however, sample from ratio should satisfy sum(ratio) = 1
    recipe = OrderedDict()
    assert (task_config is None) or (sample_from_ratio is None), f"Provide task_config or sample_from_ratio, not both."
    dataset_list = []
    dataset_files = os.listdir(data_path)
    if task_config is not None:
        task_and_ratio = task_config.split("/")
        for tar in task_and_ratio:
            name, ratio = tar.split(":")
            recipe[name] = float(ratio)
            
        for task_name, ratio in recipe.items():
            file_mask = [MAP2FILEMAME[task_name] in file_name for file_name in dataset_files]
            position = file_mask.index(True)
            this_data_path = os.path.join(data_path, dataset_files[position])
            task_dataset = NAME2DATASET[task_name](
                this_data_path,
                tokenizer,
                add_selfies,
                for_test,
                data_file=None,
                if_smiles=if_smiles
            )
            
            task_dataset, _ = random_split(task_dataset, [ratio, 1-ratio])
            dataset_list.append(task_dataset)
            
    elif sample_from_ratio is not None:
        assert total_size is not None, "When you provided sample_from_ratio, you should provide a total_size"
        task_and_prop = task_config.split("/")
        for tap in task_and_prop:
            name, proportion = tap.split(":")
            recipe[name] = float(proportion)
        assert sum([v for v in recipe.values()]) == 1.0, f"the mixture ratio should sum to 1"
        
        dataset_list = []
        dataset_files = os.listdir(data_path)
        for task_name, proportion in recipe.items():
            file_mask = [MAP2FILEMAME[task_name] in file_name for file_name in dataset_files]
            position = file_mask.index(True)
            this_data_path = os.path.join(data_path, dataset_files[position])
            task_dataset = NAME2DATASET[task_name](
                this_data_path,
                tokenizer,
                add_selfies,
                for_test,
                data_file=None,
                if_smiles=if_smiles
            )
            size = total_size * proportion
            if size > len(task_dataset):
                logger.warning(
                    f"The size under proportion {proportion} is {size}, which is greater than dataset size {len(task_dataset)}" +
                    f"will use the whole dataset, this will result in total size less than the setting."
                    )
            else:
                task_dataset, _ = random_split(task_dataset, [size, len(task_dataset)-size])
                
            dataset_list.append(task_dataset)
    else:
        raise ValueError(f"Did you forget dataset argumnts?")
    
    train_dataset = None
    val_dataset = None
    if split_val:
        val_dataset = []
        train_dataset = []
        for dataset in dataset_list:
            train, val = random_split(dataset, [1-val_ratio, val_ratio])
            train_dataset.append(train)
            val_dataset.append(val)
    else:
        train_dataset = dataset_list
    
    logger.info(f"======Mixed dataset information======", on_rank0=True)
    for idx, task in enumerate(recipe.keys()):
        logger.info(f"\033[34m{f'(Train) {task}':<25}\033[0m {f'size: {len(train_dataset[idx])}':<15} \033[34m{f'(val) {task}':<25}\033[0m {f'size: {len(val_dataset[idx]) if val_dataset is not None else 0}':<15}", on_rank0=True)
        
    logger.info(f"Total train size: {sum([len(dataset) for dataset in train_dataset])}", on_rank0=True)
    logger.info(f"Total val size: {sum([len(dataset) for dataset in val_dataset]) if val_dataset is not None else 0}", on_rank0=True)
    
    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset) if val_dataset is not None else None
    
    data_module = {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": GraphDatasetCollator(tokenizer=tokenizer)
    }
    
    return data_module

