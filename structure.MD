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
    Clearer sections:
        - LeaseTermClauses
            - startDate
            - endDate
            - nonCancellablePeriod
            - renewalOption
            - reasonablyExpectedToExercise
            - earlyTermination
        - DiscountRate
            - implicitRate
            - incrementalBorrowingRate
        - ExemptionRules
            - shortTermLease
            - lowValueAsset
            - variableLeaseOnly
        - CostSeparation
            - leaseComponent
            - nonLeaseComponent
    Ambigious sections(Extract and also evaluate):
        - PaymentTerms
            - periodicRent
            - frequency
            - timing
            - prepayment
            - finalPayment
            - variableRent
            - variableRentPredictable
                :Extract into tuple or object with structure as:　[duration(months): , Price:　], then track if it is rated 
        - isSpecificAsset: extract subject asset
            IFRS16 第 B13 至 B20 段。
        - hasControlRight: 
            IFRS16 第 B21 至 B23 段。
            IFRS16 第 B25 至 B30 段
            考量 IFRS16 第 B24 段(b)(ii)。
        - assetReplaceable
            考量 IFRS16 第 B24 段(b)(i)。
        - finalDecision
    3. calulations and balancing
        - 邏輯一致性檢查 
            - 檢查租期、付款金額、折現率等關鍵數據間的邏輯一致性
        - 金額異常檢測
            - 檢測異常的租金金額、折現率、總付款額等財務數據
        - 租期異常檢測
            - 檢查租期計算、續租條件、提前終止條款的合理性
        -    條款異常檢測
            - 檢測合約條款的完整性、一致性及與IFRS 16要求的符合度

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
