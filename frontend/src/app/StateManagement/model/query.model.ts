export interface QueryModel {
    question: string;
    isComparative: boolean;
    objects: string[];
    aspects: string[];
    numberOfArgumentsPerObject: number;
}