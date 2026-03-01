export declare function buildTokenizer(docs: string[]): {
    uchars: string[];
    BOS: number;
    vocabSize: number;
};
export declare function tokenize(doc: string, uchars: string[], BOS: number): number[];
