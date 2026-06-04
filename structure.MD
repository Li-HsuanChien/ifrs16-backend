## File Structure
    Util files grouped with features
    Configuration file to plug in ocr, llm apis
    llm prompt files llmcalls.json
      - for improvable prompts
      - file content structure
```json
  {
    "task": <title for this llm input object>,
    "description": <description for this llm input object>,
    "system_prompts": <input system prompts>,
    "rules": <ground truth based information>,
    "examples": <oneshot output example>,
    "outputSchema": <json schema for output structure> 
}
```



## TODO:
    API Call Feature: write a skeleton for api calls
    LLM Extraction calls API feature: Write a prompt 
    Extraction calls
    Calculations
    Evaluation calls from calculations
    Fix frontend data drill
    Fix frontend features:
        - local storage to view
        - allow multi step payment

##
    Frontend endpoints:
        Upload file

## Workflow:
    1. PDF/DOCX/DOC to text
        - PDF, DOCX, DOC transfer to text or img
        - img preprocessing
        - OCR to text
    2. Extract and also evaluate details for following into Typescript specified structure:
        Sample data:
```json
        "leaseJudgment": {
    "isSpecificAsset": {
      "value": true,
      "content": "合約中明確指定標的資產為生產線機器編號 ML-2025-001，供應商不得在租賃期間隨意替換其他資產，亦無替換對其具經濟利益的權利。",
      "confidence": 95,
      "userConfirmed": false
    },
    "hasControlRight": {
      "value": true,
      "content": "承租人於租賃期間有權取得該資產產生的大部分經濟效益，包括產出、收入及成本節省，並得限制其他任何人（包括出租人）使用或享有該資產之效益。",
      "confidence": 90,
      "userConfirmed": false
    },
    "assetReplaceable": {
      "value": true,
      "content": "承租人得決定資產的使用方式及用途；即便使用方式於設計時已固定，承租人仍可控制資產的運作方式及核心用途，具備對該資產之實質控制權。",
      "confidence": 85,
      "userConfirmed": false
    },
    "finalDecision": {
      "isLease": true,
      "reason": "根據IFRS 16標準，此合約符合租賃定義",
      "confidence": 90
    }
  },
  "leaseTerms": {
    "startDate": {
      "value": "2024-01-01T00:00:00.000Z",
      "content": "租賃期間自2024年1月1日起",
      "confidence": 95,
      "userConfirmed": false
    },
    "endDate": {
      "value": "2026-12-31T00:00:00.000Z",
      "content": "至2026年12月31日止",
      "confidence": 95,
      "userConfirmed": false
    },
    "nonCancellablePeriod": {
      "value": 36,
      "content": "36個月不可撤銷期間",
      "confidence": 90,
      "userConfirmed": false
    },
    "renewalOption": {
      "hasOption": true,
      "period": 24,
      "content": "承租人有權選擇續租2年",
      "confidence": 80,
      "userConfirmed": false
    },
    "reasonablyExpectedToExercise": {
      "value": true,
      "reason": "基於市場條件和承租人意圖分析",
      "confidence": 75,
      "userConfirmed": false
    },
    "earlyTermination": {
      "hasClause": false,
      "penalty": null,
      "penaltyType": null,
      "content": "合約中未發現提前終止條款",
      "confidence": 85,
      "userConfirmed": false
    },
    "prepayment": {
      "hasPrepayment": false,
      "amount": null,
      "content": "無預付款項",
      "confidence": 80,
      "userConfirmed": false
    },
    "finalPayment": {
      "hasPayment": false,
      "amount": null,
      "content": "無期末尾款",
      "confidence": 80,
      "userConfirmed": false
    }
  },
  "paymentTerms": [{
    "periodicRent": {
      "amount": 50000,
      "content": "第一年至第三年新台幣50,000元",
      "confidence": 95,
      "userConfirmed": false
    },
    "frequency": {
      "value": "monthly",
      "content": "每月繳納",
      "confidence": 90,
      "userConfirmed": false
    },
    "timing": {
      "value": "advance",
      "content": "每月1日前繳納",
      "confidence": 90,
      "userConfirmed": false
    },
    "periodStartDate": {
      "value": "2024-01-01T00:00:00.000Z",
      "content": "第一年至第三年",
      "confidence": 95,
      "userConfirmed": false
    },
    "periodEndDate": {
      "value": "2026-12-31T00:00:00.000Z",
      "content": "第一年至第三年",
      "confidence": 95,
      "userConfirmed": false
    },
    
    "variableRent": {
      "hasVariable": false,
      "type": null,
      "content": "無變動租金條款",
      "confidence": 85,
      "userConfirmed": false
    },
    "variableRentPredictable": {
      "isPredictable": null,
      "reason": "不適用",
      "confidence": 0,
      "userConfirmed": false
    }
    },{
    "periodicRent": {
      "amount": 51500,
      "content": "第四年至第六年新台幣51,500元",
      "confidence": 95,
      "userConfirmed": false
    },
    "frequency": {
      "value": "monthly",
      "content": "每月繳納",
      "confidence": 90,
      "userConfirmed": false
    },
    "timing": {
      "value": "advance",
      "content": "每月1日前繳納",
      "confidence": 90,
      "userConfirmed": false
    },
    "periodStartDate": {
      "value": "2027-01-01T00:00:00.000Z",
      "content": "第四年至第六年",
      "confidence": 95,
      "userConfirmed": false
    },
    "periodEndDate": {
      "value": "2029-12-31T00:00:00.000Z",
      "content": "第四年至第六年",
      "confidence": 95,
      "userConfirmed": false
    },
    "variableRent": {
      "hasVariable": false,
      "type": null,
      "content": "無變動租金條款",
      "confidence": 85,
      "userConfirmed": false
    },
    "variableRentPredictable": {
      "isPredictable": null,
      "reason": "不適用",
      "confidence": 0,
      "userConfirmed": false
    }
  }],
  "costSeparation": {
    "leaseComponent": {
      "amount": 47000,
      "percentage": 94,
      "allocationMethod": "relative_standalone_price",
      "contractReference": "第5條 - 租金明細",
      "confidence": 85,
      "userConfirmed": false
    },
    "nonLeaseComponent": {
      "amount": 3000,
      "percentage": 6,
      "allocationMethod": "relative_standalone_price",
      "contractReference": "第7條 - 服務條款",
      "confidence": 80,
      "userConfirmed": false
    }
  },
  "discountRate": {
    "incrementalBorrowingRate": {
      "rate": 0.045,
      "source": "市場利率分析",
      "currency": "TWD",
      "confidence": 80,
      "userConfirmed": false
    },
    "implicitRate": {
      "rate": null,
      "source": "合約中未披露",
      "currency": "TWD",
      "confidence": 0,
      "userConfirmed": false
    }
  }
```     
    
    4. endpoint with large package
    
## CICD development plan
    - modules development
        - OCR module
        - LLM extraction module
        - Evaluation module
        - Calculation verification module
        - api call module
    - integration

    - improvement
        - further period calculation
        - OCR preprocess for accuracy
        - LLM finetune for accuracy
        - Fullstack communication optimization
