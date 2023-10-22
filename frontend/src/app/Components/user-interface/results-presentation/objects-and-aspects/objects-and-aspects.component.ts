import {Component} from '@angular/core';
import {AppStateService} from "../../../../StateManagement/app-state.service";
import {Aspect} from "../../../../StateManagement/model/aspect.model";
import {map} from "rxjs";

@Component({
    selector: 'app-objects-and-aspects',
    templateUrl: './objects-and-aspects.component.html',
    styleUrls: ['./objects-and-aspects.component.css']
})
export class ObjectsAndAspectsComponent {

    constructor(public state: AppStateService) {
        this.state.state$.subscribe(state => {
            this.object_A = state.objectOne;
            this.object_B = state.objectTwo;
            this.aspects = state.aspects.map(aspect => ({value: aspect}));
        });
    }

    viewState = this.state.state$.pipe(
        map(state => state.viewState)
    );

    object_A = ''; // the first object currently entered
    object_B = ''; // the second object currently entered
    aspects: Aspect[] = [{value: ''}];
    fastSearch: boolean = false;
    numberOfArgumentsPerObject = 10;
    showResult = false; // boolean that checks if the result table should be shown


    reportObjectsAndAspects() {
        this.state.queryArgumentsAndSources([this.object_A, this.object_B], this.aspects.map(aspect => aspect.value), this.numberOfArgumentsPerObject, this.fastSearch);
    }

    forceReportObjectsAndAspects() {
        this.state.forceProcessQuestionWithObjectsAndAspects(
            this.object_A,
            this.object_B,
            this.aspects.map(aspect => aspect.value),
            this.fastSearch
        )
    }

    /**
     * Adds an aspect to the list of currently shown aspects.
     *
     */
    addAspect() {
        this.aspects.push({value: ''});
        console.log(this.aspects)
    }

    /**
     * Removes an aspect from the list of currently shown aspects which makes the UI remove this
     * aspect row.
     *
     * @param aspect the aspect row to be removed, given as a number
     */
    removeAspect(aspect: number) {
        this.aspects.splice(aspect, 1);
        if (this.aspects.length === 0) {
            this.addAspect();
        }
    }

    /**
     * Checks if the user entered something in both the first and the second object fields.
     *
     * @returns true, if the user entered something in both fields, false if not
     */
    objectsEntered() {
        return this.object_A !== '' && this.object_B !== '';
    }

    resetInput() {
        this.object_A = '';
        this.object_B = '';
        this.aspects = [{value: ''}];
        this.fastSearch = false;
    }
}
