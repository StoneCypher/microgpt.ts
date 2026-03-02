export declare function buildTokenizer(docs: string[]): {
    uchars: string[];
    BOS: number;
    vocabSize: number;
};
export declare function tokenize(doc: string | undefined, uchars: string[], BOS: number): number[];
